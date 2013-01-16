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
int iterations;
int iterationsLeft;
double errLimit;

int realSize;
double* u1;
double* u2;

double threshold;
int iterations = 0;

char mode = 'i';

double updateMat(double* from, double* to)
{
	double err = 0;
	#pragma omp for
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[i*realSize + j-1] + from[(i-1)*realSize + j] + from[i*realSize+j+1] + from[(i+1)*realSize + j] +  h*h * f(i,j,n) )*0.25;
				#pragma omg critical
				{
			err += fabs( step - to[i*realSize+j] ); }
			//printf("step %f \n ", step);
			to[i*realSize + j] = step;
		}				
	}
	#pragma omg critical
{	iterations++; }
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
	#pragma omp parallel
	{
		while((mode == 'i' && --iterationsLeft > 0)||
			(mode == 'e' && err > errLimit))
		{
			err =  updateMat(u1, u2);
			swap(&u1, &u2);		
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
