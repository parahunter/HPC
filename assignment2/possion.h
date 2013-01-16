#include <stdlib.h>
#include <stdio.h>

const double wallVal = 20.0;
const double radiatorVal = 200.0;


void print(double *u, int nn)
{
	for(int i=0; i<nn; i++)
	{
		for(int j=0; j<nn; j++)
			printf("%f\t",u[i*nn+j]);
		printf("\n");
	}
}

int c(double a, int n)
{
	return (int)( 0.5 * (double)n + 0.5 * a * (double) n );
}

int fc_1_3, fc_m2_3, fc_0, fc_m1_3;

double f(int i, int j, int n)
{
	int realSize = n + 2;	

	if(fc_0 < i && i <= fc_1_3  && fc_m2_3 < j && j <= fc_m1_3 )
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
		result[realSize * i+(realSize-1)] 	= wallVal;
		result[(realSize-1)*realSize + i] 	= wallVal;
		result[realSize * i] 			= 0;

	}
	
	fc_0=c(0,n);
	fc_1_3 = c(1.0/3.0,n);
	fc_m2_3=c(-2.0/3.0,n);
	fc_m1_3= c(-1.0/3.0,n);
	
/*
	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
			result[i*realSize+j]=f(i,j,n);
*/
	return result;
}


