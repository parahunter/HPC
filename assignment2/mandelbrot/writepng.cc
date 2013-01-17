#include <stdio.h>
#include <float.h>
#include <math.h>

#include "pngwriter.h"
#include "writepng.h"

double maxcolours = 65535;

double
maxval(int *vector, int len) {

    int i;
    double max = FLT_MIN;

    for(i = 0; i < len; i++)
	max = vector[i] > max ? vector[i] : max;

    return(max);
}

double
maxval(double *vector, int len) {

    int i;
    double max = FLT_MIN;

    for(i = 0; i < len; i++)
	max = vector[i] > max ? vector[i] : max;

    return(max);
}

void
writepng(char *filename, double *array, int x, int y) {
    
    int i, j;
	double max = maxval(array, x*y);
	double freq = 1/max;

    // create the PNGwriter object
    pngwriter png1(x ,y , (int)maxcolours, filename);

    for(i = 0; i < x; i++) {
	for(j = 0; j < y; j++) {
	    double b = array[i*y + j]*freq;
	    png1.plot(i+1, j+1, (double)b, 0.0, (double)(1-b));
	}
    }

    //  Remember to close this object - saves it to disk!!
    png1.close(); 
}

void
writepng(char *filename, int *array, int x, int y) {
    
    int i, j;
    double scale = 1.0 / maxval(array, x*y);
    double c;

    // create the PNGwriter object
    pngwriter png1(x ,y , (int)maxcolours, filename);

    for(i = 0; i < x; i++) {
	for(j = 0; j < y; j++) {
	    c = (maxcolours * sqrt((float)array[i*y + j] * scale));
	    // plot expects pixel numbers from 1..xmax, 1..ymax!
	    png1.plot(i+1, j+1, (int) (3*c/5), (int) (3*c/5), (int) c);
	}
    }

    //  Remember to close this object - saves it to disk!!
    png1.close(); 
}
