// gauss sequential
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
inline double gaussStep()
{
	double err=0;
	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
		{
			double step = (u[i*nn+j+1]+u[i*nn+j-1]+
				 u[(i+1)*nn+j]+u[(i-1)*nn+j] - h*h*f(i,j,n))*0.25;
			err += abs(step-u[i*nn+j]);
			u[i*nn+j]=step;
		}
	return err;
}

double lastErr=0;
int lastIteration=0;

inline void gaussIterations(int iterations)
{
	for(int i=0; i<iterations; i++)
		lastErr=gaussStep();
	lastIteration = iterations;
}
inline void gaussErr(double err)
{
	lastIteration=0;
	while(gaussStep()<err) lastIteration++;
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
	if(mode=='i')
	{
		gaussIterations(iteration);
	}
	else
	{
		gaussErr(errLimit);
	}

	/*
	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
			u[i*nn+j]=f(i,j);
*/

	//output section
	
	printf("Thresold:\t%f\n",lastErr);
	printf("Thresold:\t%f\n",lastErr);
	
	
	if(argc>=4 && argv[3][0]=='p')
		print(u);
}
