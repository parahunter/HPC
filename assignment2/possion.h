#include <stdlib.h>
#include <stdio.h>

const double wallVal = 20.0;
const double radiatorVal = 200.0;

//these holds the boundary conditions for the radiator
int rxMin, rxMax, ryMin, ryMax;

void print(double *u, int nn)
{
	for(int i=0; i<nn; i++)
	{
		for(int j=0; j<nn; j++)
			printf("%f\t",u[i*nn+j]);
		printf("\n");
	}
}


int cx(double a, int n)
{
	return  (0.5 + 0.5 * a ) * (double)n;
}

int cy(double a, int n)
{
	return (0.5 + 0.5 * a ) * (double)n;
}

double f(int i, int j, int n)
{
	int realSize = n + 2;	

	if(rxMin < i && i <= rxMax && ryMin < j && j <= ryMax )
		return radiatorVal;
	else
		return 0;

}


double* createMat(int n)
{
	double* result;
	int realSize = n + 2;
	
	if( (result = (double*)calloc(realSize*realSize, sizeof(double)) ) == NULL)
	{
		printf("noooo we ran out of memory or the system is mean to us :-(");
	}
	int i;

	for(i = 0 ; i < realSize -1 ; i++)
	{
		result[0 + i] 	      			= wallVal;
		result[realSize * i + (realSize-1)] 	= 0.0;
		result[(realSize-1)*realSize + i] 	= wallVal;
		result[realSize * i] 			= wallVal;

	}

	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
			result[i*realSize+j]=f(i,j,n);

	//because the image lib outputs pictures in column first order we need to change the coordinates of the radiator
	rxMin = cx(1.0/3.0,n);
	rxMax = cx(2.0/3.0,n);
	ryMin = cy(0,n);
	ryMax = cy(1.0/3.0,n);

	printf(" [%d %d] [%d %d]", rxMin, rxMax, ryMin, ryMax);

	return result;
}


