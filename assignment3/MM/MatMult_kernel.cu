//
// kernel routine
//
#include "AtomicAdd.h"
#include "stdio.h"
void MatMult_gold(const double* A, const double* x, double* y, int M, int N)
//
// Naive version where only global memory and automatic variables are accessed.
//
{
	int i, j;
	double tmp;
	for (i=0; i < N; i++) y[i] = 0;
	for (j=0; j < M; j++) {
		tmp = x[j];
		for (i=0; i < N; i++) {
			y[i] += A[i+j*N] * tmp;

		}
	}
}

extern "C" {
#include <cblas.h>
}

void MatMult_blas(const double* A, const double* x, double* y, int M, int N)
//
// Transposed matrix-vector multiplication using BLAS on CPU
//
{
	cblas_dgemv(CblasRowMajor,CblasTrans,M,N,1.0,A,N,x,1,0.0,y,1);
}

__global__ void MatMult_kernel_v1(const double* A, const double* x, double* y, int M, int N)
//
// Naive version where only global memory and automatic variables are accessed.
//

 // YOUR TASKS:
 // - Write a naive kernel where every thread compute one element of y.
 // - All global memory reads should be coalesced.
 // - Make sure that the kernel does not read or write outside memory allocated.
 //
{
	int index = (blockIdx.x * blockDim.x + threadIdx.x); 	
	
	double sum = 0.0;
	for(int i = 0; i < M; i++) {
		const double xv = x[i];
		for(int j = index; j < N; j+=gridDim.x * blockDim.x) {	
			sum += A[i*M + j]*xv;
		}
	}
	y[index] = sum;
}

#include "AtomicAdd.h"
__global__ void MatMult_kernel_v2(const double* A, const double* x, double* y, int M, int N)
//
// 2D grid + atomic add kernel
//
 // YOUR TASKS:
 // - Improve your naive kernel to support higher occupancy for wide matrices.
 // - You should use a 2D grid (of 1D thread blocks).
 // - Have threads update the output vector by using the supplied double atomic add.
 // - Use cudaMset to clear the output vector before the kernel is called.
{
//	int tid = threadIdx.x;
//	int blkidx = blockIdx.x * gridDim.x + blockIdx.y;
//	int index = (blkidx * blockDim.x + tid); 	

	int index = (blockIdx.x * blockDim.x + threadIdx.x); 	

	double sum = 0.0;
	int ysub = M / gridDim.y;
	int offset = blockIdx.y * ysub;
	for(int i = 0; i < ysub; i++) {
		const double xv = x[i + offset];
		for(int j = index; j < N; j+= blockDim.x * gridDim.x) {	
			sum += A[(i + offset)*M + j]*xv;
		}
	}
	atomicAdd(&y[index], sum);
//	y[index] = sum;
}

extern "C" {
#include <cublas.h>
}

void MatMult_cublas(const double* d_A, const double* d_x, double* d_y, int M, int N)
//
// Transposed matrix-vector multiplication using CUBLAS on GPU
//
 // YOUR TASKS:
 // - Insert a call to the function cublasDgemv() in the CUBLAS library.
{
	cublasDgemv ('N', 
			M, 
			N, 
			1.0, 
			d_A, 
			M, 
			d_x,
			1, 
			0.0, 
			d_y, 
			1);


}
