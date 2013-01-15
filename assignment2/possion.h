#include <stdlib.h>
#include <stdio.h>

const double wallVal = 25.0;

void createMat(int n, double* result)
{
	int realSize = n + 2;
	
	if( (result = calloc(realSize*realSize, sizeof(double)) ) == NULL)
	{
		printf("noooo");
	}


	printf("hello");
	int i;
	for(i = 0 ; i < realSize ; i++)
	{
		result[0 + i] 						= wallVal;
		//result[(realSize-1)*realSize + i] 	= wallVal;
		//result[realSize * i] 				= wallVal;
		//result[(realSize-1) + realSize*i] 	= 0;
	}
}




