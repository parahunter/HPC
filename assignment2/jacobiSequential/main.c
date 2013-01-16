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

double updateMat(double* from, double* to)
{
	double err = 0;
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[j*realSize + i-1] + from[(j-1)*realSize + i] + from[j*realSize+i+1] + from[(j+1)*realSize + i] +  h*h * f(i,j,n) )*0.25;
			
			err += fabs( step - to[i*realSize+j] );
			//printf("step %f \n ", step);
			to[j*realSize + i] = step;
		}				
	}	

	iterations++;

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
	printf("Jacobi Sequential\n");
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
	
	while(1)
	{
		if(mode == 'i' && iterationsLeft <= 0)
			break;
		else if(mode == 'e' && err < errLimit)
			break;

		err =  updateMat(u1, u2);

		swap(&u1, &u2);

		iterationsLeft--;		
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
