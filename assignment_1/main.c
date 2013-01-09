#include <stdio.h>
#include <stdlib.h>



#include "lib.c"

int
main ( int argc, char *argv[] )   {
	int n=3;
	int m=2;
	int k=5;
	
	double* a;
	double* b;
	double* c;
	if ( (a = calloc( m*k, sizeof(double) )) == NULL ) {
	  perror("main(__LINE__), allocation failed");
	  exit(1);
  }

	if ( (b = calloc( k*n, sizeof(double) )) == NULL ) {
	  perror("main(__LINE__), allocation failed");
	  exit(1);
  }
  
  
	if ( (c = calloc( m*n, sizeof(double) )) == NULL ) {
	  perror("main(__LINE__), allocation failed");
	  exit(1);
  }
  for(int i = 0; i<m; i++)    
    for(int j=0; j<k; j++)
    {
      a[i*k+j]=10*i+j;
    }
  for(int i = 0; i<k; i++)    
    for(int j=0; j<n; j++)
      b[i*n+j]=20*i+j;
  print(m,k,a);
  print(k,n,b);
//  mult_matrix(n,m,k,a,b,c);
  /*dgemm('N', 'N', n, m, 
         k,  1, &b[0], n, &a[0],
         k, 0, &c[0], n);*/
  print(m,n,c);
	printf("Done with my job.\n");
        return(EXIT_SUCCESS);
}
