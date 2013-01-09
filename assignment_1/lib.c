#include <sunperf.h>
#include <stdio.h>

void print(int r, int c, double* m)
{
    printf("\n---------------------\n\n");
 for(int i = 0; i<r; i++)    
 {
    for(int j=0; j<c; j++)
    {
      printf("%G\t",m[i*c+j]);
    }
    printf("\n");
  }
}

void matmult_lib(int m,int n,int k,double *A,double *B,double *C)
{
      dgemm('N', 'N', n, m, 
         k,  1, &B[0], n, &A[0],
         k, 0, &C[0], n);

}


/*
Starting permutation of n-m-k
How many possibilities?watch out for init
*/
void matmult_kmn(int m,int n,int k,double *A,double *B,double *C)
{
  for(int j=0; j<m; j++)
    for(int i = 0; i<n; i++)  
    {
      C[j*n+i]=0;
    }
  for(int l=0; l<k; l++)
    for(int j=0; j<m; j++)
      for(int i = 0; i<n; i++)  

        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_knm(int m,int n,int k,double *A,double *B,double *C)
{
  for(int j=0; j<m; j++)
    for(int i = 0; i<n; i++)  
    {
      C[j*n+i]=0;
    }
  for(int l=0; l<k; l++)
    for(int i = 0; i<n; i++)  
      for(int j=0; j<m; j++)
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_mnk(int m,int n,int k,double *A,double *B,double *C)
{
  for(int j=0; j<m; j++)
    for(int i = 0; i<n; i++)  
    {
      C[j*n+i]=0;
      for(int l=0; l<k; l++)
      {
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
      }
    }
}
void matmult_nmk(int m,int n,int k,double *A,double *B,double *C)
{
  for(int i = 0; i<n; i++)  
    for(int j=0; j<m; j++)
    {
      C[j*n+i]=0;
      for(int l=0; l<k; l++)
      {
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
      }
    }
}
void matmult_nat(int m,int n,int k,double *A,double *B,double *C)
{
  for(int j=0; j<m; j++)
    for(int i = 0; i<n; i++)  
    {
      C[j*n+i]=0;
      for(int l=0; l<k; l++)
      {
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
      }
    }
}

void matmult_blk_internal(int m,int n,int k, int sm,int sn,int sk,double *A,double *B,double *C, int bs)
{
  for(int j=sm*bs; j<(sm+1)*bs; j++)
    for(int i = sn*bs; i<(sn+1)*bs; i++)  
      for(int l=sk*bs; l<(sk+1)*bs; l++)
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs)
{
  print(m,k,A);
  print(k,n,B);
  for(int j=0; j<m; j++)
    for(int i = 0; i<n; i++)  
    {
      C[j*n+i]=0;
    }

  for(int j=0; j<m/bs; j+=bs)
    for(int i = 0; i<n/bs; i+=bs)  
    {
      for(int l=0; l<k/bs; l+=bs)
      {
        matmult_blk_internal(m,n,k,j,i,l,A,B,C,bs);
        print(m,n,C);
      }
    }
}



