//jacobi omp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "../possion.h"
#include "../image.c"

int n = 0;
double h,hh;

int iterationsLeft;
double errLimit;

int realSize;
double* u1;
double* u2;

double threshold;
int iterations = 0;
double err;

char mode = 'i';

void updateMat(double* from, double* to)
{
	#pragma omp parallel for schedule(runtime)
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[i*realSize + j-1] + from[(i-1)*realSize + j] + from[i*realSize+j+1] + from[(i+1)*realSize + j] +  hh * f(i,j,n) )*0.25;
			to[i*realSize + j] = step;
		}
	}
}
void updateMatE(double* from, double* to)
{
	#pragma omp parallel for schedule(runtime) reduction(+: err)
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[i*realSize + j-1] + from[(i-1)*realSize + j] + from[i*realSize+j+1] + from[(i+1)*realSize + j] +  hh * f(i,j,n) )*0.25;
			double te = fabs( to[i*realSize+j] - from[i*realSize+j] );

			{	err += te; }
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



int main ( int argc, char *argv[] ) 
{
	printf("Jacobi Sequential\n");
	if(argc>=2)
		n = atoi(argv[1]);
	else
		n = 6;

	realSize = n + 2;
	h = 2.0/n;
	hh=h*h;
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
	
	
	if(mode=='i')
	{
		iterations=iterationsLeft;
		
		{
		for(int i=0; i<(iterationsLeft/2)-1; i++)
		{
			updateMat(u1, u2);
			updateMat(u2, u1);
		}
		updateMat(u1, u2);

		{ err = 0; }
		updateMatE(u2, u1);
		}
	}
	if(mode=='e')
	{
		err = 2*errLimit;
		iterations=0;
		int iterBlock=100;
		
		{
		while(err>errLimit)
		{

			for(int i=0; i<(iterBlock/2)-1; i++)
			{
				updateMat(u1, u2);
				updateMat(u2, u1);
			}
			updateMat(u1, u2);
			
			{ err=0; iterations += iterBlock;}
			updateMatE(u2, u1);
		}
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
