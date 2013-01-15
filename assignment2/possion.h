#include <stdlib.h>
#include <stdio.h>

const double wallVal = 20.0;

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




