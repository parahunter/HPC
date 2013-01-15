#include <stdlib.h>
#include <stdio.h>

const double wallVal = 20.0;
const double radiatorVal = 200.0;

double* createMat(int n)
{
	double* result;
	int realSize = n + 2;
	
	if( (result = calloc(realSize*realSize, sizeof(double)) ) == NULL)
	{
		printf("noooo we ran out of memory or the system is mean to us :-(");
	}
	int i;
	for(i = 0 ; i < realSize -1 ; i++)
	{
		result[0 + i] 						= wallVal;
		result[(realSize-1)*realSize + i] 	= wallVal;
		result[realSize * i] 				= wallVal;
	}

	return result;
}

int c(double a, int n)
{
	return (int)( 0.5 * (double)n + 0.5 * a * (double) n );
}

double f(int i, int j, int n)
{
	int realSize = n + 2;	

	if(c(0,n) < i && i <= c(1.0/3.0,n)  && c(-2.0/3.0,n) < j && j <= c(-1.0/3.0,n) )
		return radiatorVal;
	else
		return 0;

}


