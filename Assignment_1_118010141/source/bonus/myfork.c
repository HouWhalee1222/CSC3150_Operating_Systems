#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

void info2txt(int SIG, char strpid[], char strppid[], int normal){
	FILE *fp = NULL;
	fp = fopen("siginfo.txt","a+");
	fputs("The child process(pid=",fp);fputs(strpid,fp);
	fputs(") of parent process(pid=",fp);fputs(strppid,fp);fputs(") ",fp);

	if (normal){ // normal
		char normalstr[5];
		fputs("has normal execution\nIts exit status = ",fp);
		sprintf(normalstr,"%d",SIG);
		fputs(normalstr,fp);fputs("\n",fp);
	}
	else{ // signal
		char sigstr[5];
		fputs("is terminated by signal\n",fp);
		sprintf(sigstr,"%d",SIG);
		fputs("Its signal number = ",fp);
		fputs(sigstr,fp);fputs("\n",fp);

		switch (SIG){
		case SIGABRT:{
			fputs("child process get SIGABRT signal\n",fp);
			fputs("child process is abort by abort signal\n",fp);
			break;
		}
		case SIGALRM:{
			fputs("child process get SIGALRM signal\n",fp);
			fputs("child process releases alarm signal\n",fp);
			break;
		}
		case SIGBUS:{
			fputs("child process get SIGBUS signal\n",fp);
			fputs("child process has bus error\n",fp);
			break;
		}
		case SIGFPE:{
			fputs("child process get SIGFPE signal\n",fp);
			fputs("child process has Floating point exception\n",fp);
			break;
		}
		case SIGHUP:{
			fputs("child process get SIGHUP signal\n",fp);
			fputs("child process is hung up\n",fp);
			break;
		}
		case SIGILL:{
			fputs("child process get SIGILL signal\n",fp);
			fputs("child process has illegal instruction\n",fp);
			break;
		}				
		case SIGINT:{
			fputs("child process get SIGINT signal\n",fp);
			fputs("child process has teminal interrupt\n",fp);
			break;
		}
		case SIGKILL:{
			fputs("child process get SIGKILL signal\n",fp);
			fputs("child process is killed\n",fp);
			break;
		}				
		case SIGPIPE:{
			fputs("child process get SIGPIPE signal\n",fp);
			fputs("child process writes on a pipe with no reader, Broken pipe\n",fp);
			break;
		}
		case SIGQUIT:{
			fputs("child process get SIGQUIT signal\n",fp);
			fputs("child process has terminal quit\n",fp);
			break;
		}
		case SIGSEGV:{
			fputs("child process get SIGSEGV signal\n",fp);
			fputs("child process has invalid memory segment access\n",fp);
			break;
		}
		case SIGTERM:{
			fputs("child process get SIGTERM signal\n",fp);
			fputs("child process terminates\n",fp);
			break;
		}
		case SIGTRAP:{
			fputs("child process get SIGTRAP signal\n",fp);
			fputs("child process reach a breakpoint\n",fp);
			break;
		}
		default:{
			fputs("no matching signals\n",fp);
			break;
		}
		}
	}
	fputs("\n",fp);
	fclose(fp);
}


void execute(int arg_len, char *arg_list[], int first){
	char strpid[10];
	char strchid[10];
	
	if (arg_list[1]==NULL){ // only one program in list
		FILE *fp = NULL;
		fp = fopen("pid.txt","a+");
		sprintf(strpid,"%d",getpid());
		fputs(strpid,fp);fputs("\n",fp);
		fclose(fp);

		execve(arg_list[0],arg_list,NULL);
	}

	else{

		pid_t pid;
		int status;
		int i;
		char* arg[arg_len];
		
		
		// change the program list to execute
		for (int i=0;i<arg_len;i++){
			arg[i] = arg_list[i+1];
		}
		arg[arg_len-1] = NULL;

		pid = fork();
		if (pid==-1){
			perror("fork");
			exit(1);
		}

		else{

			if (pid==0){ // child - do next recursion
				raise(SIGCHLD);
				execute(arg_len, arg, 0);
			}

			else{ // parent - execute current program
			
				waitpid(pid, &status, WUNTRACED);

				FILE *fp = NULL;
				fp = fopen("pid.txt","r");
				while (!feof(fp)){
					fgets(strchid,10,fp); // get the pid of child process
				}
				strchid[strlen(strchid)-1] = '\0'; // delete \n
				fclose(fp);

				fp = fopen("pid.txt","a+");
				sprintf(strpid,"%d",getpid());
				fputs(strpid,fp);fputs("\n",fp);
				fclose(fp);

				
				if (WIFEXITED(status)){
					info2txt(WEXITSTATUS(status),strchid,strpid,1);
				}
				else if (WIFSIGNALED(status)){
					info2txt(WTERMSIG(status),strchid,strpid,0);
				}
				else{
					printf("CHILD PROCESS CONTINUED\n");
				}

				if (first){ // whether it is myfork function
					FILE *fp = NULL;
					fp = fopen("siginfo.txt","a+");
					fputs("Myfork process(pid=",fp);
					fputs(strpid,fp);
					fputs(") execute normally\n",fp);
					fclose(fp);
				}
				else execve(arg_list[0],arg,NULL); // a bug here		
			}
		}
	}
	return;
}

void init(){
	FILE *fp = NULL;
	fp = fopen("pid.txt","w");
	fclose(fp);
	fp = fopen("siginfo.txt","w");
	fclose(fp);
}

void printinfo(int argc){
	int i;
	char line[50][10], info[50];
	FILE *fp = NULL;
	
	printf("the process tree : ");
	fp = fopen("pid.txt","r");
	for (int i=0;i<argc;i++){
		fgets(line[i],10,fp);
		line[i][strlen(line[i])-1] = '\0'; // delete \n
	}
	for (int i=0;i<argc;i++){
		printf("%s",line[argc-i-1]);
		if (i<argc-1) printf("->");
	}
	printf("\n");
	fclose(fp);

	fp = fopen("siginfo.txt","r");
	while (!feof(fp)){
		memset(info,0,sizeof(info));
		fgets(info,50,fp);
		printf("%s",info);
	}
	fclose(fp);
}


int main(int argc,char *argv[]){

	if (argc<=1){
		perror("No program to execute\n");
		exit(1);
	}

	init();

	execute(argc, argv, 1);

	printinfo(argc);	

	return 0;
}