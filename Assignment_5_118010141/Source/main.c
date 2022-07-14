#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <linux/interrupt.h>
#include <asm/uaccess.h>
#include <asm/io.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

// DEVICE
#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region
#define IRQ_NUM 1

static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;

/* pointer to dma_buffer */
void *dma_buf;

/* interrupt number */
int interrupt_count;

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2



// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo);
static int drv_open(struct inode* ii, struct file* ff);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo);
static int drv_release(struct inode* ii, struct file* ff);
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

// Interrupt handler
irq_handler_t irq_handler(int irq, void *dev_id, struct pt_regs *regs);

// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


// File operations
static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

/* Read operation for the device */
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	int ans;
	ans = myini(DMAANSADDR);
	printk("%s:%s():ans = %d", PREFIX_TITLE, __func__, ans);
	put_user(ans, (int*) buffer);
	myouti(0, DMAANSADDR); // clean the result
	myouti(0,DMAREADABLEADDR); // set readable as false
	return 0;
}

/* Write operation for the device */
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	int IOMode;

	IOMode = myini(DMABLOCKADDR);
	dataIn = kmalloc(sizeof(*dataIn), GFP_KERNEL);
	// printk("%s:%s(): IO Mode is %d\n", PREFIX_TITLE, __func__, IOMode);
	 __copy_from_user(dataIn, buffer, (unsigned long) ss); 
	myoutc(dataIn->a, DMAOPCODEADDR);
	myouti(dataIn->b, DMAOPERANDBADDR);
	myouts(dataIn->c, DMAOPERANDCADDR);
	printk("%s:%s: queue work",PREFIX_TITLE, __func__);
	INIT_WORK(work_routine, drv_arithmetic_routine);

	// Decide io mode
	if(IOMode==1) {
		// Blocking IO
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();
    	} 
	else if (IOMode==0){
		// Non-Blocking IO
		// printk("%s:%s(): non-blocking\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
   	}
	return 0;
}


/* Ioctl setting for the device */
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	int value;
	int readable;
	readable = 0;

	get_user(value, (int *) arg);

	switch (cmd)
	{
	case HW5_IOCSETSTUID:
		myouti(value, DMASTUIDADDR);
		printk("%s:%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, myini(DMASTUIDADDR));
		break;

	case HW5_IOCSETRWOK:
		myouti(value, DMARWOKADDR);
		printk("%s:%s(): RW (read/write operation) OK!\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETIOCOK:
		myouti(value, DMAIOCOKADDR);
		printk("%s:%s(): IOC (ioctl function) OK!\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETIRQOK:
		myouti(value, DMAIRQOKADDR);
		printk("%s:%s(): IRC (interrupt service routine) OK!\n", PREFIX_TITLE, __func__);
		break;

	case HW5_IOCSETBLOCK:
		if (value==1){
			printk("%s:%s(): Blocking IO\n", PREFIX_TITLE, __func__);
		}
		else if (value==0){
			printk("%s:%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		}
		myouti(value, DMABLOCKADDR);
		break;

	case HW5_IOCWAITREADABLE:
		readable = myini(DMAREADABLEADDR);
		while (!readable){
			msleep(5000);
			readable = myini(DMAREADABLEADDR);
		}
		printk("%s:%s(): wait readable %d\n", PREFIX_TITLE, __func__, readable);
		put_user(readable, (int *) arg);
		break;

	default:
		printk("%s:%s(): Wrong command!\n", PREFIX_TITLE, __func__);
		break;
	}
	return 0;
}



/* Arthemetic routine */
static void drv_arithmetic_routine(struct work_struct* ws) {
	char operator;
	int operand1, operand2;
	int ans;

	// Variables for prime function
	int fnd;
	int i, num, isPrime;

	operator = myinc(DMAOPCODEADDR);
	operand1 = myini(DMAOPERANDBADDR);
	operand2 = myins(DMAOPERANDCADDR);
	ans = 0;

	switch (operator){
	case '+':
		ans = operand1 + operand2;
		break;
	
	case '-':
		ans = operand1 - operand2;
		break;
	case '*':
		ans = operand1 * operand2;
		break;
	case '/':
		ans = operand1 / operand2;
		break;
	case 'p':
		fnd = 0;
		num = operand1;
		while (fnd != operand2){
			isPrime = 1;
			num++;
			for (i=2;i<=num/2;i++){
				if (num%i==0){
					isPrime = 0;
					break;
				}
			}

			if (isPrime) fnd++;
		}
		ans = num;
		break;
	default:
		printk("%s:%s(): Wrong command!\n", PREFIX_TITLE, __func__);
		break;
	}

	printk("%s:%s(): %d %c %d = %d", PREFIX_TITLE, __func__, operand1, operator, operand2, ans);
	myouti(ans, DMAANSADDR);

	// Set the readable as true
	myouti(1, DMAREADABLEADDR);
}

irq_handler_t irq_handler(int irq, void *dev_id, struct pt_regs *regs){
	interrupt_count++;
	return (irq_handler_t) IRQ_HANDLED;
}


static int __init init_modules(void) {
    
	dev_t dev;
	int ret;

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
	
	/* Initialize interrupt module and register IRQ handler */ 
	interrupt_count = 0;
	ret = request_irq(IRQ_NUM, (irq_handler_t) irq_handler, IRQF_SHARED, DEV_NAME, (void *)(irq_handler));
	printk("%s:%s(): request_irq %d return %d", PREFIX_TITLE, __func__, IRQ_NUM, ret);
	
	/* Register chrdev */
  	ret = alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME);

	// Register chrdev
	if(ret < 0) { // ...
		printk(KERN_ALERT"Register chrdev failed!\n");
		return ret;
    } 
	
    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);  
	printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);
    
	dev_cdev = cdev_alloc();
	if (dev_cdev==NULL){
		printk(KERN_ALERT"Fail to allocate!\n");
		return -1;
	}

	/* Init cdev and make it alive */
	cdev_init(dev_cdev, &fops);
	dev_cdev->owner = THIS_MODULE;
	dev_cdev->ops = &fops; 

	ret = cdev_add(dev_cdev, dev, DEV_COUNT);
	if(ret < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return ret;
   	}

	/* Allocate DMA buffer */
	dma_buf = kmalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s(): allocate dma buffer\n",PREFIX_TITLE, __func__);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);


	return ret;
}

static void __exit exit_modules(void) {

	dev_t dev;
	dev = MKDEV(dev_major, dev_minor);

	printk("%s:%s(): interrupt count=%d\n",PREFIX_TITLE, __func__, interrupt_count);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s(): free dma buffer\n",PREFIX_TITLE, __func__);

	/* Delete character device */
	cdev_del(dev_cdev);

	unregister_chrdev_region(dev, DEV_COUNT);
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);

	/* Free work routine */
	kfree(work_routine);

	/* Free IRQ */
	free_irq(IRQ_NUM, (void *)(irq_handler));

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);