#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){
	
	pid_t pid;
	int status;

	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();

	if (pid==-1){ // fail to fork
		perror("fork");
		exit(1);
	}
	else{

		// Child Process
		if (pid==0){
			int i;
			char* arg[argc];
			
			printf("I'm the child process, my pid = %d\n",getpid());
			// raise(SIGCHLD);

			/* execute test program */ 
			printf("Child process start to execute the program\n");
			

			for (int i=0;i<argc-1;i++){
				arg[i] = argv[i+1]; // agrv-arg
			}
			arg[argc-1] = NULL;
			execve(arg[0],arg,NULL);

			// Exception
			perror("execve");
			exit(EXIT_FAILURE);
			
		}
		
		// Parent Process
		else{
			printf("I'm the parent process, my pid = %d\n",getpid());
			
			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);
			printf("Parent process receiving the SIGCHLD signal\n");
			
			/* check child process'  termination status */
			if (WIFEXITED(status)){
				printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
			}
			else if (WIFSIGNALED(status)){

				switch (WTERMSIG(status)){
				case SIGABRT:{
					printf("child process get SIGABRT signal\n");
					printf("child process is abort by abort signal\n");
					break;
				}
				case SIGALRM:{
					printf("child process get SIGALRM signal\n");
					printf("child process releases alarm signal\n");
					break;
				}
				case SIGBUS:{
					printf("child process get SIGBUS signal\n");
					printf("child process has bus error\n");
					break;
				}
				case SIGFPE:{
					printf("child process get SIGFPE signal\n");
					printf("child process has Floating point exception\n");
					break;
				}
				case SIGHUP:{
					printf("child process get SIGHUP signal\n");
					printf("child process is hung up\n");
					break;
				}
				case SIGILL:{
					printf("child process get SIGILL signal\n");
					printf("child process has illegal instruction\n");
					break;
				}				
				case SIGINT:{
					printf("child process get SIGINT signal\n");
					printf("child process has teminal interrupt\n");
					break;
				}
				case SIGKILL:{
					printf("child process get SIGKILL signal\n");
					printf("child process is killed\n");
					break;
				}				
				case SIGPIPE:{
					printf("child process get SIGPIPE signal\n");
					printf("child process writes on a pipe with no reader, Broken pipe\n");
					break;
				}
				case SIGQUIT:{
					printf("child process get SIGQUIT signal\n");
					printf("child process has terminal quit\n");
					break;
				}
				case SIGSEGV:{
					printf("child process get SIGSEGV signal\n");
					printf("child process has invalid memory segment access\n");
					break;
				}
				case SIGTERM:{
					printf("child process get SIGTERM signal\n");
					printf("child process terminates\n");
					break;
				}
				case SIGTRAP:{
					printf("child process get SIGTRAP signal\n");
					printf("child process reach a breakpoint\n");
					break;
				}
				default:{
					printf("no matching signals\n");
					break;
				}
				} 
				printf("CHILD EXECUTION FAILED!!\n");
			}

			else if (WIFSTOPPED(status)){
				printf("child process get SIGSTOP signal\n");
				printf("child process stopped\n");
				printf("CHILD PROCESS STOPPED\n");
			}
			else{
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}	
	return 0;
}