//jacobi sequential

#include "../possion.h"
#include <stdio.h>

int n = 0;
double h;
int iterations;
double errLimit;

int realSize;
double* u1;
double* u2;

double threshold;
int iterations;
double time;

char mode = 'i';

double updateMat(double* from, double* to)
{
	double err = 0;
	for(int i = 1 ; i < realSize -1; i++)
	{
		for(int j = 1 ; j < realSize -1; j++)
		{
			double step = (from[i*realSize + j-1] + from[(i-1)*realSize + j] +
						   from[i*realSize+j+1]   + from[(i+1)*realSize + j] +
						   h*h * f(i,j,n) )*0.25;
			
			err += abs( step - to[i*realSize+j] );
			//printf("step %f \n ", step);
			to[i*realSize + j] = step;
		}				
	}	

	return err;
}

void print()
{
	for(int i = 0 ; i < realSize; i++)
	{
		for(int j = 0 ; j < realSize ; j++)
		{
			printf("%f ", u1[i*realSize + j]);
		}
		printf("\n");
	}
}

void swap(double** a, double** b)
{
	double* temp = *a;
	*a = *b;
	*b = temp;
}

int main ( int argc, char *argv[] ) 
{
	
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
			iterations = atoi(argv[3]);
		}
			else 
		if(argv[2][0] == 'e')
		{
			mode = 'e';
			errLimit = atof(argv[3]);
		}
	}

	while(1)
	{
		double err =  updateMat(u1, u2);
		printf("error %f \n", err);

		swap(&u1, &u2);

		if(argc>=4)
			if(argv[4][0] == 'p')
				print();		

		if(mode == 'i' && --iterations < 0)
			break;
		else if(mode == 'e' && err < errLimit)
			break;
	}	
	
	return 0;
}
