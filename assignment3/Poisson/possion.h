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
	n-=2;
	return 1+(int)( 0.5 * (double)n + 0.5 * a * (double) n );
}

int fc_1_3, fc_m2_3, fc_0, fc_m1_3;

double f(int i, int j)
{


	if(fc_0 < i && i <= fc_1_3  && fc_m2_3 < j && j <= fc_m1_3 )
		return radiatorVal;
	else
		return 0;

}


void initMat(double *mat, int n)
{
	for(int i = 0 ; i < n - 1 ; i++)
	{
		mat[0 + i] 	      			= wallVal;
		mat[n * i+(n-1)] 	= wallVal;
		mat[(n-1)*n + i] 	= wallVal;
		mat[n * i] 			= 0;

	}
	
	fc_0=c(0,n);
	fc_1_3 = c(1.0/3.0,n);
	fc_m2_3=c(-2.0/3.0,n);
	fc_m1_3= c(-1.0/3.0,n);
	
}


