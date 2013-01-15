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
    double   *image;

    width    = 2601;
    height   = 2601;
    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    image = (double *)malloc( width * height * sizeof(double));
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }
	/*
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
	*/
	for(int i = 0; i < width; i++) {
		for(int j = 0; j < height; j++) {
			image[i*height + j] = (double)j;
		}
	}

	//start = omp_get_wtime();
    writepng("mandelbrot.png", image, width, height);
	/*end = omp_get_wtime() ;
	double write = end-start;
	printf("Serial mand  : %lf\n", sermand);
	printf("Parallel mand: %lf\n", parmand);
	printf("PNG write    : %lf\n", write);
	*/
    return(0);
}
