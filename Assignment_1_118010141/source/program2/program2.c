#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

typedef unsigned long ul;

static struct task_struct* task;

struct wait_opts {
	enum pid_type wo_type;
	int	wo_flags;
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	int __user *wo_stat;
	struct rusage __user *wo_rusage;
	wait_queue_t child_wait;
	int	notask_error;
};


extern long _do_fork(unsigned long clone_flags,
	      unsigned long stack_start,
	      unsigned long stack_size,
	      int __user *parent_tidptr,
	      int __user *child_tidptr,
	      unsigned long tls);
extern int do_execve(struct filename* filename,
		  const char __user *const __user *__argv,
          const char __user *const __user *__envp);
extern struct filename* getname(const char __user * filename);
extern long do_wait(struct wait_opts *wo);

// implement exec function
int my_exec(void){
	int result;
	// const char path[] = "/home/seed/Downloads/program2/test";
	const char path[] = "/home/seed/work/assignment1/source/program2/test";
	const char *const argv[] = {path,NULL,NULL};
	const char *const envp[] = {"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};

	struct filename* my_filename = getname(path);

	/* execute a test program in child process */
	result = do_execve(my_filename, argv, envp);

	// if succeed
	if (!result) return 0;
	// if fail
	do_exit(result);
}

// implement wait function
void my_wait(pid_t pid){
	int status;
	struct wait_opts wo;
	struct pid* wo_pid = NULL;
	enum pid_type type;
	long a;
	bool normal = false;
	
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED; // think
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*)&status;
	wo.wo_rusage = NULL;

	a = do_wait(&wo); // return value of do_wait

	// for stop

	switch (*wo.wo_stat){
	case SIGABRT:{
		printk("[program2] : child process get SIGABRT signal\n");
		printk("[program2] : child process is abort by abort signal\n");
		break;
	}
	case SIGALRM:{
		printk("[program2] : child process get SIGALRM signal\n");
		printk("[program2] : child process releases alarm signal\n");
		break;
	}
	case SIGBUS:{
		printk("[program2] : child process get SIGBUS signal\n");
		printk("[program2] : child process has bus error\n");
		break;
	}
	case SIGFPE:{
		printk("[program2] : child process get SIGFPE signal\n");
		printk("[program2] : child process has Floating point exception\n");
		break;
	}
	case SIGHUP:{
		printk("[program2] : child process get SIGHUP signal\n");
		printk("[program2] : child process is hung up\n");
		break;
	}
	case SIGILL:{
		printk("[program2] : child process get SIGILL signal\n");
		printk("[program2] : child process has illegal instruction\n");
		break;
	}				
	case SIGINT:{
		printk("[program2] : child process get SIGINT signal\n");
		printk("[program2] : child process has teminal interrupt\n");
		break;
	}
	case SIGKILL:{
		printk("[program2] : child process get SIGKILL signal\n");
		printk("[program2] : child process is killed\n");
		break;
	}				
	case SIGPIPE:{
		printk("[program2] : child process get SIGPIPE signal\n");
		printk("[program2] : child process writes on a pipe with no reader, Broken pipe\n");
		break;
	}
	case SIGQUIT:{
		printk("[program2] : child process get SIGQUIT signal\n");
		printk("[program2] : child process has terminal quit\n");
		break;
	}
	case SIGSEGV:{
		printk("[program2] : child process get SIGSEGV signal\n");
		printk("[program2] : child process has invalid memory segment access\n");
		break;
	}
	case SIGTERM:{
		printk("[program2] : child process get SIGTERM signal\n");
		printk("[program2] : child process terminates\n");
		break;
	}
	case SIGTRAP:{
		printk("[program2] : child process get SIGTRAP signal\n");
		printk("[program2] : child process reach a breakpoint\n");
		break;
	}
	default:{
		// judge stop
		if (*wo.wo_stat>>8==SIGSTOP){
			printk("[program2] : child process get SIGSTOP signal\n"); //bug
			printk("[program2] : child process stopped\n");
		}
		else{
			normal = true;
			printk("[program2] : Normal termination\n");
		} 
		break;
	}
	} 

	// output child process exit status
	if ((*wo.wo_stat>>8)==SIGSTOP){
		printk("[program2] : The return signal is %d\n", SIGSTOP);
	}
	else{
		if (normal) *wo.wo_stat = *wo.wo_stat >> 8;
		printk("[program2] : The return signal is %d\n", *wo.wo_stat);
	}
	
	put_pid(wo_pid);

	return;
}


//implement fork function
int my_fork(void *argc){
	
	//set default sigaction for current process
	int i;
	long pid;
	struct k_sigaction *k_action = &current->sighand->action[0];
	
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD,(ul)&my_exec,0,NULL,NULL,0); // child process's pid
	printk("[program2] : The child process has pid = %ld\n", pid);
	printk("[program2] : This is the parent process, pid = %d\n", (int)current->pid);

	/* wait until child process terminates */
	my_wait((pid_t)pid);

	return 0;
}

static int __init program2_init(void){

	printk("[program2] : Module_init\n");
	
	/* create a kernel thread to run my_fork */
	printk("[program2] : Module_init create kthread start\n");
	task = kthread_create(&my_fork, NULL, "MyThread");

	// wake up new thread if it's fine
	if (!IS_ERR(task)){
		printk("[program2] : Module_init kthread start\n");
		wake_up_process(task);
	}
	
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);