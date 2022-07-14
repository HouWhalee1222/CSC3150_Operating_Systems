#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
	for (int i = 0; i < input_size; i++) {
		vm_write(vm, i, input[i]);
		printf("%d\n", i);
	}

	printf("Page fault: %d\n", *vm->pagefault_num_ptr);
	for (int i = input_size - 1; i >= input_size - 32769; i--) { // 32KB + 1
		int value = vm_read(vm, i);
		printf("%d\n", i);
	}
	printf("Page fault: %d\n", *vm->pagefault_num_ptr);
	vm_snapshot(vm, results, 0, input_size);
} 
