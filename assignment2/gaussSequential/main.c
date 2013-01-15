// gauss sequential
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//#include "writepng.cc"
#include "../possion.h"


int n, nn;
double h;
double *u;



void print(double *u)
{
	for(int i=0; i<nn; i++)
	{
		for(int j=0; j<nn; j++)
			printf("%f\t",u[i*nn+j]);
		printf("\n");
	}
}


// Return: error!
double gaussStep()
{
	double err=0.0;
	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
		{
			double step = (u[i*nn+j+1]+u[i*nn+j-1]+
				 u[(i+1)*nn+j]+u[(i-1)*nn+j] - h*h*f(i,j,n))*0.25;
			err += fabs(step-u[i*nn+j]);
			u[i*nn+j]=step;
		}
	return err;
}

double lastErr=0;
int lastIteration=0;

void gaussIterations(int iterations)
{
	for(int i=0; i<iterations; i++)
		lastErr=gaussStep();
	lastIteration = iterations;
}
void gaussErr(double err)
{
	lastIteration=1;
	while((lastErr = gaussStep())>err) lastIteration++;
}
int main(int argc, char* argv[])
{
	printf("Gauss Sequential\n");
	n=5;
	if(argc>=2) n = atoi(argv[1]);
	printf("N=%i\n",n);
	nn = n+2;
	h = 2.0/n;

	char mode='i'; // e for Err or i for iteration
	int iterations;
	double errLimit;
	if(argc>=4) 
	{
		if(argv[2][0] == 'i')
		{
			mode = 'i';
			iterations = atoi(argv[3]);
		}
		if(argv[2][0] == 'e')
		{
			mode = 'e';
			errLimit = atof(argv[3]);
		}
	}

	u=createMat(n);
	double wt = omp_get_wtime();
	clock_t t = clock();
	if(mode=='i')
	{
		gaussIterations(iterations);
	}
	else
	{
		gaussErr(errLimit);
	}
	wt = omp_get_wtime()-wt;
	t = clock()-t;

	//output section
	
	printf("Thresold:\t%f\n",lastErr);
	printf("Iterations:\t%i\n",lastIteration);
	printf("W Time:\t%f\n",wt);
	printf("C Time:\t%f\n",((float)t)/CLOCKS_PER_SEC);
	//writeimg(n,u);
	if(argc>=5 && argv[4][0]=='p')
		print(u);
}
