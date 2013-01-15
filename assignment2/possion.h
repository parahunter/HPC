#include <stdlib.h>
#include <stdio.h>

const double wallVal = 20.0;
const double radiatorVal = 200.0;



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
/*
	for(int i=1; i<=n; i++)
		for(int j=1; j<=n; j++)
			result[i*realSize+j]=f(i,j,n);
*/
	return result;
}
/*
void writeimg(int n, double *u) {

	for(int i = 0; i < (n+2)*(n+2); i++) {
			u[i]=u[i]/200.0;
	}
  writepng("img.png", u, n, n);
}*/



