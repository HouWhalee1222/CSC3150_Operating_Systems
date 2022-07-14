#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;

__device__ __managed__ u32 empty_bit = 0; // The first available empty bit

// FCB structure: 
// filename (20bytes)
// modifytime (2 bytes)
// createtime (2 bytes)
// filesize (4 bytes)
// location (4 bytes)

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}


// Judge whether two strings are the same
__device__ bool compare_file_name(uchar *first, char *second){
  while (*first!='\0' && *second!='\0'){
    if (*first!=*second) return false;
    first++;
    second++;
  }
  if (*first=='\0' && *second=='\0') return true;
  return false;
} 

__device__ u32 fs_open(FileSystem *fs, char *s, int op){
  /* Implement open operation here */
  uchar *fcb_start = fs->volume + fs->SUPERBLOCK_SIZE;
  int empty_entry = -1; // The first empty FCB entry!!!
  bool file_exist = false;
  u32 fp; // read/write on first bit + fcb index

  if (op==G_READ){
    fp = 0; // read 
  }
  else if (op==G_WRITE){
    fp = 1024; // write
  }

  
  // Find the file name in FCB
  for (int i=0;i<fs->FCB_ENTRIES;i++){  
    // Find the first empty entry in FCB
    uchar *fcb_cur = fcb_start + i * fs->FCB_SIZE;

	if (empty_entry==-1 && *fcb_cur == '\0') {
		empty_entry = i;
	}
    // Check the file name
    if (compare_file_name(fcb_cur,s)==true){
      file_exist = true;
      fp += i;
      break;
    }
  }

  // If the file is found, return its location
  if (file_exist){
    return fp;
  }
  else{ // not found
    if (op==G_WRITE){ // write mode: create a new zero byte file
      // write the information into empty entry
      uchar *fcb_new = fcb_start + empty_entry * fs->FCB_SIZE;
	    short *create_time = (short*)(fcb_new + 22);
      u32 *file_size = (u32*)(fcb_new + 24);
      u32 *location = (u32*)(fcb_new + 28);
      
      // Fill in the FCB information
      for (char *c=s;*c!='\0';c++){
          *fcb_new = *c;
		  fcb_new++;
      }
      *fcb_new = '\0'; // copy the file name
      // strcpy(empty_entry, s);

      gtime++;
      *create_time = gtime;
      *file_size = 0;

	  
      // Find an empty bit in super control block
      for (int i=0;i<fs->SUPERBLOCK_SIZE;i++){
        uchar uc = *(fs->volume + i);
        bool ex = false;
        for (int j=7;j>=0;j--){
          uchar uc_bit = (uc>>j) & 1;
          if (uc_bit==0){
            *location = i * 8 + 7 - j;
            ex = true;
            break;
          }
        }
        if (ex==true) break;
      }
	 
	  fp += empty_entry; // fp - fcb number, not location
    }
	

    else if (op==G_READ){ // read mode: error?
      printf("The read file to open does not exist\n");
    }
  }
  return fp;
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
  /* Implement read operation here */
  u32 mode = (fp >= 1024);
  u32 fcb_index = fp % fs->FCB_ENTRIES;
  
  if (mode==G_WRITE){ // wrong mode
    printf("Wrong mode when reading a file\n");
  }

  uchar *fcb_start = fs->volume + fs->SUPERBLOCK_SIZE;
  uchar *fcb_cur = fcb_start + fcb_index * fs->FCB_SIZE;
  u32 *location = (u32*)(fcb_cur + 28);
  
  uchar *read_start = fcb_start + fs->FCB_SIZE * fs->FCB_ENTRIES + fs->STORAGE_BLOCK_SIZE * (*location);
  for (int i=0;i<size;i++){
    *output = *read_start;
    read_start++;
    output++;
  }
}

// Smallest unit: block !!!

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
  /* Implement write operation here */
  u32 mode = (fp >= 1024);
  u32 fcb_index = fp % fs->FCB_ENTRIES;

  // Information from FCB
  uchar *fcb_start = fs->volume + fs->SUPERBLOCK_SIZE;
  uchar *fcb_cur = fcb_start + fcb_index * fs->FCB_SIZE; // bug!!!
  short *modify_time = (short*)(fcb_cur + 20);
  u32 *file_size = (u32*)(fcb_cur + 24);
  u32 *location = (u32*)(fcb_cur + 28);

  // How many bits the new file occupied
  u32 bit_num_new = ceil(double(size)/fs->STORAGE_BLOCK_SIZE);
  u32 bit_num_ori = ceil(double(*file_size)/fs->STORAGE_BLOCK_SIZE);
  int bit_diff = bit_num_new - bit_num_ori;
  if (bit_diff < 0) bit_diff = -bit_diff;

  if (mode==G_READ){ // wrong mode
    printf("Wrong mode when writing a file\n");
    return 0;
  }

  gtime++;
  *modify_time = gtime; // update the time

  // Case 1: write a new file
  // update FCB, file content, bit-map

  if (*file_size==0){
    
    uchar *write_start = fcb_start + fs->FCB_SIZE * fs->FCB_ENTRIES + fs->STORAGE_BLOCK_SIZE * (*location);
    uchar *bit_start = fs->volume + (*location) / 8;
    uchar bit_offset = (*location) % 8;

    *file_size = size; // update fcb (size)

    for (int i=0;i<size;i++){ // update file content
    // write the file into the storage
      *write_start = *input;
      write_start++;
      input++; 
    }

    for (int i=0;i<bit_num_new;i++){ // update bit map
      *bit_start += (1<<(7-bit_offset));
      bit_offset++;
      if (bit_offset==8){
        bit_offset -= 8;
        bit_start++;
      }
    }

	empty_bit += bit_num_new;
  }
  
  // Case 2: write into an old file, compact first, then write
  // update FCB, file content, bit-map 
  else{

    u32 compact_offset = bit_num_ori * fs->STORAGE_BLOCK_SIZE; // how many bytes to move
    int fcb_arr[1024]; // record the index for bubble sort
    int len = 0; int temp;

    uchar *file_content_start = fcb_start + fs->FCB_SIZE * fs->FCB_ENTRIES;
    u32 file_location_last = 0;
    u32 file_size_last = *(u32*)(fcb_start + 24);


    for (int i=0;i<fs->FCB_ENTRIES;i++){ // traverse the FCB, find the location after current file
      uchar *fcb_cur = fcb_start + i * fs->FCB_SIZE;
      u32 *fcb_location = (u32*)(fcb_cur + 28);

      if (*fcb_location>*location){ // if the file needs to be compacted
        fcb_arr[len] = i;
        len++;
      }
    }

    // bubble sort the fcb according to their location
    for (int i=0;i<len-1;i++){
      for (int j=0;j<len-i-1;j++){
        u32 *fcb_location_pre = (u32*)(fcb_start + fcb_arr[j] * fs->FCB_SIZE + 28);
        u32 *fcb_location_nxt = (u32*)(fcb_start + fcb_arr[j+1] * fs->FCB_SIZE + 28);
        if (*fcb_location_pre > *fcb_location_nxt){
          temp = fcb_arr[j];
          fcb_arr[j] = fcb_arr[j+1];
          fcb_arr[j+1] = temp;
        }
      }
    }

    for (int i=0;i<len;i++){
      u32 *file_location_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 28);
      u32 *file_size_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 24);
      // u32 *file_size_cur = file_location_cur - 1;
      uchar *file_content_cur = file_content_start + fs->STORAGE_BLOCK_SIZE * (*file_location_cur);
      uchar *file_content_new = file_content_cur - compact_offset;

      for (int j=0;j<*file_size_cur;j++){ // update file content
        *file_content_new = *file_content_cur;
        file_content_cur++;
        file_content_new++;
      }

      if (i==len-1){
        // file_location_last = *(file_size_cur + 1);
        // u32 file_cur_bit = ceil(double(*file_size_cur)/fs->STORAGE_BLOCK_SIZE);
        file_location_last = *file_location_cur;
        file_size_last = *file_size_cur;
      }
    }

    // update bit-map: only deal with the difference at last
	u32 file_bit_last = ceil(double(file_size_last) / fs->FCB_SIZE);
    
    if (bit_num_ori > bit_num_new){ // old > new, eliminate some bits
    
      uchar *bit_end = fs->volume + (file_location_last + file_bit_last - 1) / 8; // point to the start of a byte
	  // End of all the files originally
      uchar bit_offset = (file_location_last + file_bit_last - 1) % 8;
  
      for (int k=0;k<bit_diff;k++){
        *bit_end -= (1<<(7-bit_offset)); // ??
        if (bit_offset==0){
          bit_offset += 8;
          bit_end--;
        }
		bit_offset--;
      }

    }
    else if (bit_num_ori < bit_num_new){ // old < new, add some bits

      uchar *bit_start = fs->volume + (file_location_last + file_bit_last) / 8;
      uchar bit_offset = (file_location_last + file_bit_last) % 8;

      for (int k=0;k<bit_diff;k++){
        *bit_start += (1<<(7-bit_offset));
        bit_offset++;
        if (bit_offset==8){
          bit_offset -= 8;
          bit_start++;
        }
      }

    }

	empty_bit = empty_bit - bit_num_ori + bit_num_new;

	// update the location 
	for (int i = 0; i < len; i++) {
		u32 *fcb_location_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 28);
		*fcb_location_cur -= bit_num_ori;
	}

    // update new FCB
    *location = file_location_last - bit_num_ori + file_bit_last;
    *file_size = size; // update newly written file 
    
    // write data into the file content
    uchar *write_start = fcb_start + fs->FCB_SIZE * fs->FCB_ENTRIES + fs->STORAGE_BLOCK_SIZE * (*location);

    for (int i=0;i<size;i++){
    // write the file into the storage
      *write_start = *input;
      write_start++;
      input++;
    }
  }

  return 0; // ??
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
  /* Implement LS_D and LS_S operation here */
  uchar *fcb_start = fs->volume + fs->SUPERBLOCK_SIZE;
  // For bubble sort
  int fcb_arr[1024];
  int len = 0; int temp;

  // Calculate how many files now
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    uchar *fcb_cur = fcb_start + i * fs->FCB_SIZE;
    if (*fcb_cur!='\0'){
      fcb_arr[len] = i;
      len++;
    }
  }

  if (op==LS_D){
    
    printf("===sort by modified time===\n");


    for (int i=0;i<len-1;i++){
      for (int j=0;j<len-i-1;j++){
        short *fcb_modify_time_pre = (short*)(fcb_start + fcb_arr[j] * fs->FCB_SIZE + 20);
        short *fcb_modify_time_nxt = (short*)(fcb_start + fcb_arr[j+1] * fs->FCB_SIZE + 20);
        if (*fcb_modify_time_pre < *fcb_modify_time_nxt){
          temp = fcb_arr[j];
          fcb_arr[j] = fcb_arr[j+1];
          fcb_arr[j+1] = temp;
        }
      }
    }



    for (int i=0;i<len;i++){
      uchar *fcb_cur = fcb_start + fcb_arr[i] * fs->FCB_SIZE;
      printf("%s\n",fcb_cur);
    }
  }

  if (op==LS_S){

    printf("===sort by file size===\n");

    for (int i=0;i<len-1;i++){
      for (int j=0;j<len-i-1;j++){
        bool swap = false;
        u32 *fcb_size_pre = (u32*)(fcb_start + fcb_arr[j] * fs->FCB_SIZE + 24);
        u32 *fcb_size_nxt = (u32*)(fcb_start + fcb_arr[j+1] * fs->FCB_SIZE + 24);
        if (*fcb_size_pre < *fcb_size_nxt) swap = true;
        else if (*fcb_size_pre == *fcb_size_nxt){
		  short *fcb_create_time_pre = (short*)(fcb_start + fcb_arr[j] * fs->FCB_SIZE + 22);
          short *fcb_create_time_nxt = (short*)(fcb_start + fcb_arr[j+1] * fs->FCB_SIZE + 22);
          if (*fcb_create_time_pre > *fcb_create_time_nxt) swap = true;
        }
        if (swap){
          temp = fcb_arr[j];
          fcb_arr[j] = fcb_arr[j+1];
          fcb_arr[j+1] = temp;
        }
      }
    }

    for (int i=0;i<len;i++){
      uchar *fcb_cur = fcb_start + fcb_arr[i] * fs->FCB_SIZE;
      int file_size_cur = *(u32*)(fcb_cur + 24);
      printf("%s %d\n", fcb_cur, file_size_cur);
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  /* Implement rm operation here */
  if (op!=RM) return;

  int fcb_arr[1024];
  int len = 0; int temp;

  uchar *fcb_start = fs->volume + fs->SUPERBLOCK_SIZE;
  uchar *file_content_start = fcb_start + fs->FCB_SIZE * fs->FCB_ENTRIES;

  uchar *fcb_delete;
  u32 file_location_last = 0;
  u32 file_size_last = *(u32*)(fcb_start + 24);
  short *file_modify_time_delete;
  short *file_create_time_delete;
  u32 *file_location_delete; 
  u32 *file_size_delete; 
 
  bool file_exist = false;
  
  // Traverse the FCB: find the file to be deleted
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    uchar *fcb_cur = fcb_start + i * fs->FCB_SIZE;
  
    // Find the file to be deleted
    if (compare_file_name(fcb_cur,s)==true){
      file_exist = true; 
      fcb_delete = fcb_start + i * fs->FCB_SIZE;
	  file_modify_time_delete = (short*)(fcb_delete + 20);
	  file_create_time_delete = (short*)(fcb_delete + 22);
      file_size_delete = (u32*)(fcb_delete + 24);
      file_location_delete = (u32*)(fcb_delete + 28);
    }
  }

  if (!file_exist) return;

  u32 bit_num_delete = ceil(double(*file_size_delete)/fs->STORAGE_BLOCK_SIZE);
  u32 compact_offset = bit_num_delete * fs->STORAGE_BLOCK_SIZE;

  // Traverse the FCB: record the files after deleted file
  for (int i=0;i<fs->FCB_ENTRIES;i++){
    uchar *fcb_cur = fcb_start + i * fs->FCB_SIZE;
    u32 *fcb_location = (u32*)(fcb_cur + 28);
    
    if (*fcb_location > *file_location_delete){
      fcb_arr[len] = i;
      len++;
    }
  }

  // bubble sort the fcb according to their location
  for (int i=0;i<len-1;i++){
    for (int j=0;j<len-1-i;j++){
      u32 *fcb_location_pre = (u32*)(fcb_start + fcb_arr[j] * fs->FCB_SIZE + 28);
      u32 *fcb_location_nxt = (u32*)(fcb_start + fcb_arr[j+1] * fs->FCB_SIZE + 28);
      if (*fcb_location_pre > *fcb_location_nxt){
        temp = fcb_arr[j];
        fcb_arr[j] = fcb_arr[j+1];
        fcb_arr[j+1] = temp;
      }
    }
  }

  for (int i=0;i<len;i++){
    u32 *file_location_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 28);
    u32 *file_size_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 24);
    uchar *file_content_cur = file_content_start + fs->STORAGE_BLOCK_SIZE * (*file_location_cur);
    uchar *file_content_new = file_content_cur - compact_offset;

    for (int j=0;j<*file_size_cur;j++){ // update file content
      *file_content_new = *file_content_cur;
      file_content_cur++;
      file_content_new++;
    }

    if (i==len-1){
      file_location_last = *file_location_cur;
      file_size_last = *file_size_cur;
    }
  }
  
  u32 file_bit_last = ceil(double(file_size_last) / fs->STORAGE_BLOCK_SIZE);
  uchar *bit_end = fs->volume + (file_location_last + file_bit_last - 1) / 8;
  uchar bit_offset = (file_location_last + file_bit_last - 1) % 8;
  

  // update bit-map: eliminate the last several bits


  for (int i=0;i<bit_num_delete;i++){
    *bit_end -= (1<<(7-bit_offset));
    if (bit_offset==0){
      bit_offset += 8;
      bit_end--;
    }
    bit_offset--;
  }

  empty_bit -= bit_num_delete;


  // update the location
  for (int i = 0; i < len; i++) {
	  u32 *fcb_location_cur = (u32*)(fcb_start + fcb_arr[i] * fs->FCB_SIZE + 28);
	  *fcb_location_cur -= bit_num_delete;
  }

  // clear the original FCB
  while (*fcb_delete!='\0') {
    *fcb_delete = '\0';
	fcb_delete++;
  }

  *file_modify_time_delete = 0;
  *file_create_time_delete = 0;
  *file_location_delete = 0;
  *file_size_delete = 0;

}