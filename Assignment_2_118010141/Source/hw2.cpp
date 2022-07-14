#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

// Clear the screen
#define CLEAR "\033[H\033[2J"

// Some parameters
#define ROW 10
#define COLUMN 50 
#define NUM_THREAD 9
#define WIN 1
#define LOSE 2
#define QUIT -1
#define LOG_LENGTH 15

struct Node{
	int x, y;
	Node(int _x, int _y): x(_x), y(_y){};
	Node(){}; 
}frog; 


char map[ROW+1][COLUMN+1];
int stop = 0; // whether the program reaches an end
int sleep_time = 80000;

pthread_mutex_t mutex;

/*  Print the map and the hint on the screen  */
void *print_map(void *t){
	while (!stop){
		pthread_mutex_lock(&mutex);
		printf(CLEAR);
		for(int i=0;i<=ROW;i++){
			puts(map[i]);
		}
		pthread_mutex_unlock(&mutex);
		usleep(sleep_time);
	}
	pthread_exit(NULL);
}


/*  Check game's status  */
int game_status(int flag){
	printf(CLEAR);
	if (flag==QUIT){
		printf("You exit the game.\n");
	}
	else if (flag==WIN){
		printf("You win the game!!\n");
	}
	else if (flag==LOSE){
		printf("You lose the game!!\n");
	}
	return 1;
}

// Determine whether a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF){
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

// Threads for moving the logs
void *logs_move(void *t){
	
	int row = (int) t;
	srand(time(0)+row*row*row); // set a seed
	int start = rand()%COLUMN; // 0 - 49 

	/*  Move the logs  */
	
	while (!stop){
		pthread_mutex_lock(&mutex); // modify shared data map: add mutex

		for (int i=0;i<COLUMN;i++){
			map[row][i] = ' ';
		}

		if (row%2){ // odd row: move left
			for (int i=start;i>start-LOG_LENGTH;i--){
				int j = i;
				if (j<0){
					j += COLUMN;
				}
				map[row][j] = '=';
			}

			if (row==frog.x){ // if the frog is in this row
				if (map[row][frog.y]=='='){
					map[row][frog.y]='0';
					frog.y--;
					if (frog.y<0) stop = game_status(LOSE);
				}
				else stop = game_status(LOSE); // lose the game
			}

			start--;
			if (start<0){
				start += COLUMN;
			}
		}
		else{ // even row: move right
			for (int i=start;i<start+LOG_LENGTH;i++){
				int j = i;
				if (j>=COLUMN){
					j -= COLUMN;
				}
				map[row][j] = '=';
			}

			if (row==frog.x){ // if the frog is in this row
				if (map[row][frog.y]=='='){
					map[row][frog.y]='0';
					frog.y++;
					if (frog.y>COLUMN) stop = game_status(LOSE);
				}
				else stop = game_status(LOSE); // lose the game
			}

			start++;
			if (start>=COLUMN){
				start -= COLUMN;
			}
		}
		pthread_mutex_unlock(&mutex); // modification ends

		usleep(sleep_time);
	}
	pthread_exit(NULL);
}

// Threads for moving the frog
void *frog_move(void *t){
	/*  Check keyboard hits, to change frog's position or quit the game. */
	int flag = 0;

	while (!flag && !stop){
		if (kbhit()){
			char dir = getchar();
			if (dir=='w' || dir=='W'){
				frog.x--;
				if (frog.x==0) flag = WIN;
				if (frog.x!=ROW){
					map[ROW][frog.y] = '|';
				}
			}
			else if (dir=='a' || dir=='A'){
				frog.y--;
				if (frog.y==0 || frog.y==COLUMN-1) flag = LOSE;
				if (frog.x==ROW){
					map[ROW][frog.y] = '0';
					map[ROW][frog.y+1] = '|';
				}
			}
			else if (dir=='d' || dir=='D'){
				frog.y++;
				if (frog.y==0 || frog.y==COLUMN-1) flag = LOSE;
				if (frog.x==ROW){
					map[ROW][frog.y] = '0';
					map[ROW][frog.y-1] = '|';
				}
			}
			else if (dir=='s' || dir=='S'){
				frog.x++;
				if (frog.x>ROW) flag = LOSE;
				if (frog.x==ROW){
					map[ROW][frog.y] = '0';
				}
			}
			else if (dir=='q' || dir=='Q'){
				flag = QUIT;
			}
		}
	}
	/*  Check game's status  */
	if (!stop) stop = game_status(flag);

	pthread_exit(NULL);
}

// Exception case - fail to create the thread
void thread_expception(int rc){
	if (rc){
		printf("ERROR: return code from pthread_create() is %d", rc);
		exit(1);
	}
}

// Initialize the river map and frog's starting position
void init_map(){
	
	memset(map, 0, sizeof(map));

	for (int i=1;i<ROW;i++){	
		for (int j=0;j<COLUMN;j++){
			map[i][j] = ' ';  
		}
	}

	for (int j=0;j<COLUMN;j++){	
		map[ROW][j] = map[0][j] = '|';
	}

	frog = Node(ROW, (COLUMN+1)/2); 
	map[frog.x][frog.y] = '0'; 
}


int main(int argc, char *argv[]){

	// Initialize the map
	init_map();

	/*  Create pthreads for wood move and frog control.  */
	pthread_t log_threads[NUM_THREAD];
	pthread_t print_thread;
	pthread_t frog_thread;
	int rc;
	long i;

	pthread_mutex_init(&mutex, NULL);

	// create threads for log movement
	for (i=0;i<NUM_THREAD;i++){
		rc = pthread_create(&log_threads[i],NULL,logs_move,(void*)(i+1));
		thread_expception(rc);
	}

	// create a thread for printing the map & hint
	rc = pthread_create(&print_thread,NULL,print_map,(void*)(0));
	thread_expception(rc);

	// create a thread for moving the frog
	rc = pthread_create(&frog_thread,NULL,frog_move,(void*)(0));
	thread_expception(rc);

	// Join the thread
	for (int i=0;i<NUM_THREAD;i++){
		pthread_join(log_threads[i],NULL);
	}
	pthread_join(print_thread,NULL);
	pthread_join(frog_thread,NULL);

	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL); // Pthread_Termination

	return 0;

}

// g++ -o hw2 hw2.cpp -lpthread
// ./a.out