#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i=0;i<vm->PAGE_ENTRIES;i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] = 0; // Array for implementing LRU
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr, int page_count,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;
  vm->page_count = page_count;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}
  
__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  u32 page_offset = addr % vm->PAGESIZE;
  u32 page_number = addr >> 5;
  u32 frame_number;
  bool page_fault = true;

  // Find the page number in inverted-page-table
  for (int i=0;i<vm->PAGE_ENTRIES;i++){
    if (vm->invert_page_table[i] == 0x00000000){ // valid
      if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_number){
        page_fault = false;
        frame_number = i;
        // update the LRU array
        for (int j=vm->page_count-1;j>=0;j--){
          u32 entry = vm->invert_page_table[j + 2*vm->PAGE_ENTRIES];
          if (entry==page_number){ // find its original position
            for (int k=j-1;k>=0;k--){// update the previous one: move to next index one by one
              vm->invert_page_table[k + 1 + 2*vm->PAGE_ENTRIES] = vm->invert_page_table[k + 2*vm->PAGE_ENTRIES];
            }
            vm->invert_page_table[2*vm->PAGE_ENTRIES] = page_number;
            break;
          }
        }
      }
    }
  }

  // Deal with page-fault
  if (page_fault){
    u32 least_frame;
    (*vm->pagefault_num_ptr)++; // increase page_fault count
    
    // Update the page table if full
    if (vm->page_count==vm->PAGE_ENTRIES){ // The page table is full
      u32 least_entry = vm->invert_page_table[3*vm->PAGE_ENTRIES-1];
      // replace the least used page number with the current one
      for (int t=0;t<vm->PAGE_ENTRIES;t++){
        if (vm->invert_page_table[t + vm->PAGE_ENTRIES]==least_entry){
          least_frame = t; // record the least used frame number
          vm->invert_page_table[t + vm->PAGE_ENTRIES] = page_number; // update the page table
          break;
        }
      }
    }
    
    // Update the LRU array: move back one by one
    for (int i=vm->page_count-1;i>=0;i--){ 
      if (i==vm->PAGE_ENTRIES-1) continue;
      vm->invert_page_table[i + 1 + 2*vm->PAGE_ENTRIES] = vm->invert_page_table[i + 2*vm->PAGE_ENTRIES];
    }
    vm->invert_page_table[2*vm->PAGE_ENTRIES] = page_number; // The first one becomes the current page
    

    // Load the page in the secondary memory
	// Case 1: no empty frame: Swap
	if (vm->page_count == vm->PAGE_ENTRIES) {
	  uchar* swap_page_storage_out = vm->storage + page_number * vm->PAGESIZE;
	  uchar* swap_page_storage_in = vm->storage + least_frame * vm->PAGESIZE;
	  uchar* swap_page_memory = vm->buffer + least_frame * vm->PAGESIZE;
	  for (int j = 0; j < vm->PAGESIZE; j++) {
		swap_page_storage_in[j] = swap_page_memory[j];
		swap_page_memory[j] = swap_page_storage_out[j];
	  }
	  frame_number = least_frame;
	}

	//Case 2: there is still empty frame
	if (vm->page_count < vm->PAGE_ENTRIES) {
		for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
			if (vm->invert_page_table[i] == 0x80000000) {
				frame_number = i;
				vm->page_count++;
				vm->invert_page_table[frame_number] = 0x00000000; // set to valid
				vm->invert_page_table[frame_number + vm->PAGE_ENTRIES] = page_number;
				break;
			}
		}
		uchar* swap_page_storage_out = vm->storage + page_number * vm->PAGESIZE;
		uchar* swap_page_memory = vm->buffer + frame_number * vm->PAGESIZE;
		for (int j = 0; j < vm->PAGESIZE; j++) {
			swap_page_memory[j] = swap_page_storage_out[j];
		}
	}
  }

  // access the data
  uchar target_data = vm->buffer[frame_number*vm->PAGESIZE + page_offset];
  return target_data;
}

// write 1 byte each time
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) { 
  /* Complete vm_write function to write value into data buffer */
  u32 page_offset = addr % vm->PAGESIZE;
  u32 page_number = addr >> 5;
  u32 frame_number;
  bool page_fault = true;

  // Find the page number in inverted-page-table
  for (int i=0;i<vm->PAGE_ENTRIES;i++){
    if (vm->invert_page_table[i]==0x00000000){ // valid
      if (vm->invert_page_table[i + vm->PAGE_ENTRIES]==page_number){
        page_fault = false;
        frame_number = i;
        // update the LRU array
        for (int j=vm->page_count-1;j>=0;j--){
          u32 entry = vm->invert_page_table[j + 2*vm->PAGE_ENTRIES];
          if (entry==page_number){
            for (int k=j-1;k>=0;k--){
              vm->invert_page_table[k + 1 + 2*vm->PAGE_ENTRIES] = vm->invert_page_table[k + 2*vm->PAGE_ENTRIES];
            }
            vm->invert_page_table[2*vm->PAGE_ENTRIES] = page_number;
            break;
          }
        }
      }
    }
  }

  // Deal with page fault
  if (page_fault){
    u32 least_frame;
    (*vm->pagefault_num_ptr)++;
    
    // Update the page table if full
    if (vm->page_count==vm->PAGE_ENTRIES){
      u32 least_entry = vm->invert_page_table[3*vm->PAGE_ENTRIES-1];
      for (int t=0;t<vm->PAGE_ENTRIES;t++){ // find the least entry in page table
        if (vm->invert_page_table[t + vm->PAGE_ENTRIES]==least_entry){
          least_frame = t;
          vm->invert_page_table[t + vm->PAGE_ENTRIES] = page_number;
          break;
        }
      }
    }

    // Update the LRU array: Move back one by one
    for (int i=vm->page_count-1;i>=0;i--){ 
      if (i==vm->PAGE_ENTRIES-1) continue;
      vm->invert_page_table[i + 1 + 2*vm->PAGE_ENTRIES] = vm->invert_page_table[i + 2*vm->PAGE_ENTRIES];
    }
    vm->invert_page_table[2*vm->PAGE_ENTRIES] = page_number;

	// Load the page in the secondary memory
    // Case 1: no empty frame: Swap
    if (vm->page_count==vm->PAGE_ENTRIES){
	  uchar* swap_page_storage_out = vm->storage + page_number * vm->PAGESIZE;
	  uchar* swap_page_storage_in = vm->storage + least_frame * vm->PAGESIZE;
	  uchar* swap_page_memory = vm->buffer + least_frame * vm->PAGESIZE;
	  for (int j = 0; j < vm->PAGESIZE; j++) {
		swap_page_storage_in[j] = swap_page_memory[j];
		swap_page_memory[j] = swap_page_storage_out[j];
	  }
      frame_number = least_frame;
    }
    
    //Case 2: there is still empty frame
    if (vm->page_count<vm->PAGE_ENTRIES){
      for (int i=0;i<vm->PAGE_ENTRIES;i++){
        if (vm->invert_page_table[i]==0x80000000){
          frame_number = i;
          vm->page_count++;
          vm->invert_page_table[frame_number] = 0x00000000; // set to valid
          vm->invert_page_table[frame_number + vm->PAGE_ENTRIES] = page_number;
          break;
        }
      }
    }
  }

  // Write the data
  vm->buffer[frame_number*vm->PAGESIZE + page_offset] = value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (u32 i=offset;i<input_size;i++){
    uchar value = vm_read(vm, i);
	printf("%d\n", i);
    results[i] = value;
  }
}