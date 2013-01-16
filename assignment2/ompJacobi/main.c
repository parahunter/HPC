//jacobi sequential

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../possion.h"
#include "../image.c"

int n = 0;
double h;

int iterationsLeft;
double errLimit;

int realSize;
double* u1;
double* u2;

double threshold;
int iterations = 0;

char mode = 'i';

void updateMat(double* from, double* to)
{
	#pragma omp for schedule(runtime)
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[i*realSize + j-1] + from[(i-1)*realSize + j] + from[i*realSize+j+1] + from[(i+1)*realSize + j] +  h*h * f(i,j,n) )*0.25;
			to[i*realSize + j] = step;
		}
	}
}

double errCheck(double *from, double *to)
{
	double err=0;
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			err += fabs( to[i*realSize+j] - from[i*realSize+j] );
		}
	}
	return err;
}

void swap(double** a, double** b)
{
	double* temp = *a;
	*a = *b;
	*b = temp;
}



int main ( int argc, char *argv[] ) 
{
	printf("Jacobi basic Omp\n");
	if(argc>=2)
		n = atoi(argv[1]);
	else
		n = 6;

	realSize = n + 2;
	h = 2.0/n;
	u1 = createMat(n);
	u2 = createMat(n);

	if(argc>=3) 
	{
		if(argv[2][0] == 'i')
		{
			mode = 'i';
			iterationsLeft = atoi(argv[3]);
		}
			else 
		if(argv[2][0] == 'e')
		{
			mode = 'e';
			errLimit = atof(argv[3]);
		}
	}


	double wt = omp_get_wtime();
	clock_t t = clock();
	double err;
	
	
	if(mode=='i')
	{
		iterations=iterationsLeft;
			#pragma omp parallel
			{
			for(int i=0; i<iterationsLeft/2; i++)
			{

				updateMat(u1, u2);
				updateMat(u2, u1);
				}
			}
		err = errCheck(u1, u2);
	}
	if(mode=='e')
	{
		err = 2*errLimit;
		iterations=0;
		int iterBlock=1000;
		while(err>errLimit)
		{
			#pragma omp parallel
			{
			for(int i=0; i<iterBlock/2; i++)
			{
				updateMat(u1, u2);
				updateMat(u2, u1);
			}
			}
			iterations += iterBlock;
			err = errCheck(u1, u2);
		}
	}
	wt = omp_get_wtime()-wt;
	t = clock()-t;

	printf("Thresold:\t%f\n",err);
	printf("Iterations:\t%i\n",iterations);
	printf("W Time:\t%f\n",wt);
	printf("C Time:\t%f\n",((float)t)/CLOCKS_PER_SEC);
	
	writeImg (n+2, u1);

	if(argc>=5 && argv[4][0] == 'p')
		print(u1, realSize);

	return 0;
}
