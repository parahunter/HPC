//
// kernel routine
//
#include "AtomicAdd.h"
#include "stdio.h"
void MatMult_gold(const double* A, const double* B, double* C, int M, int N, int K)
//
// Naive version.
//
{

}

extern "C" {
#include <cblas.h>
}

void MatMult_blas(const double* A, const double* B, double* C, int M, int N, int K)
//
// Transposed matrix-vector multiplication using BLAS on CPU
//
{
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}

__global__ void MatMult_kernel_v1(const double* A, const double* B, double* C, int M, int N, int K)
//
// Naive version where only global memory and automatic variables are accessed.
//

 // YOUR TASKS:
 // - Write a naive kernel where every thread compute one element of y.
 // - All global memory reads should be coalesced.
 // - Make sure that the kernel does not read or write outside memory allocated.
 //
{
	//reversed order for better coalescence 
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	int index = i * N + j;
	
	//printf("(%d, %d) %d \n", i,j,index);

	double sum = 0.0;

	for(int k = 0; k < K; k++) 
	{
		sum += A[k + i * N] * B[ k*K + j];  
	}

	C[index] = sum;
}

#include "AtomicAdd.h"
__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K)
{
	const int BLOCKSIZE = 4;
	//reversed order for better coalescence 
	int jb = ( blockIdx.x * blockDim.x + threadIdx.x ) * BLOCKSIZE;
	int ib = ( blockIdx.y * blockDim.y + threadIdx.y ) * BLOCKSIZE;
	
	int index = ib * N + jb;
	
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

//	for(int i = 0 ; i < BLOCKSIZE * BLOCKSIZE ; i++)				
	//	sm[ i] = 0.0;

		for(int k = 0; k < K; k++) 
		{

			for(int j = 0 ; j < BLOCKSIZE ; j++)
			{	
	for(int i = 0 ; i < BLOCKSIZE ; i++)
	{
			

				sm[j + BLOCKSIZE * i ] += A[k + (ib + i) * N]  * - B[ k*K + (jb + j)];  
			}
		}
	}
	
	for(int i = 0 ; i < BLOCKSIZE ; i++)
	{
		for(int j = 0 ; j < BLOCKSIZE ; j++)
		{
			C[(i+ib) * N + (j+jb)] = sm[j + BLOCKSIZE * i ];
			//if(ib == 4 && jb == 8)
			//	printf("(%d, %d) %d \n", ib, jb, (i+ib) * N + (j+jb));
		}
	}
}

extern "C" {
#include <cublas.h>
}

void MatMult_cublas(const double* d_A, const double* d_B, double* d_C, int M, int N, int K)
//
// Transposed matrix-vector multiplication using CUBLAS on GPU
//
{
	cublasDgemm('N','N', M, N, K, 1.0, d_B, N, d_A, K, 0.0, d_C, N);
}
