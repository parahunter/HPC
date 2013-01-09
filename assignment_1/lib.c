#include <sunperf.h>
#include <stdio.h>

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

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
  int nm = n*m;
  for(int i = 0; i<nm; i++)  
  {
    C[i]=0;
  }
  for(int l=0; l<k; l++)
    for(int j=0; j<m; j++)
      for(int i = 0; i<n; i++)
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_knm(int m,int n,int k,double *A,double *B,double *C)
{
  int nm = n*m;
  for(int i = 0; i<nm; i++)  
  {
    C[i]=0;
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
void matmult_nkm(int m,int n,int k,double *A,double *B,double *C)
{
  int nm = n*m;
  for(int i = 0; i<nm; i++)  
  {
    C[i]=0;
  }

  for(int i = 0; i<n; i++)  
    for(int l=0; l<k; l++)
      for(int j=0; j<m; j++)
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_mkn(int m,int n,int k,double *A,double *B,double *C)
{
  int nm = n*m;
  for(int i = 0; i<nm; i++)  
  {
    C[i]=0;
  }
  for(int j=0; j<m; j++)
    for(int l=0; l<k; l++)
      for(int i = 0; i<n; i++)  
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
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
  int mmin = min(m,(sm+1)*bs);
  int nmin = min(n,(sn+1)*bs);
  int kmin = min(k,(sk+1)*bs);
  int smbs = sm*bs;
  int snbs = sn*bs;
  int skbs = sk*bs;
  for(int j=smbs; j<mmin; j++)
    for(int l=skbs; l<kmin; l++)
      for(int i = snbs; i<nmin; i++)  
        C[j*n+i]+=A[j*k+l]*B[l*n+i];
}
void matmult_blk(int m,int n,int k,double *A,double *B,double *C, int bs)
{
  //bs=50;
  int nm = n*m;
  for(int i = 0; i<nm; i++)  
  {
    C[i]=0;
  }
  int m_bs = m/bs;
  int n_bs = n/bs;
  int k_bs = k/bs;
  for(int j=0; j<=m_bs; j++)
    for(int l=0; l<=k_bs; l++)
      for(int i = 0; i<=n_bs; i++)    
        matmult_blk_internal(m,n,k,j,i,l,A,B,C,bs);
}
