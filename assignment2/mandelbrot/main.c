#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#ifdef _OPENMP
#include <omp.h>
#endif

int
main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int  *image;

    width    = 2600;
    height   = 2600;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    image = (int *)malloc( width * height * sizeof(int));
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }
	
	double start, end;

	start = omp_get_wtime();
	mandel(width, height, image, max_iter);
	end = omp_get_wtime() ;
    double sermand = end - start;

	start = omp_get_wtime();
	#pragma omp parallel shared(image)
	{
		mandel(width, height, image, max_iter);
	}	
	end = omp_get_wtime() ;
	double parmand = end - start;


	start = omp_get_wtime();
    writepng("mandelbrot.png", image, width, height);
	end = omp_get_wtime() ;
	double write = end-start;
	printf("Thresold:\t%lf\n", sermand);
	printf("Iterations:\t%lf\n", sermand);
	printf("WTime: \t%lf\n", parmand);
	printf("Ctime   :\t%lf\n", write);
	
    return(0);
}
