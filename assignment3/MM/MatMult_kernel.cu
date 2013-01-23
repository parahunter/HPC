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
/*	int index = (blockIdx.x * blockDim.x + threadIdx.x); 	
	
	double sum = 0.0;
	for(int i = 0; i < M; i++) {
		const double xv = x[i];
		for(int j = index; j < N; j+=gridDim.x * blockDim.x) {	
			sum += A[i*M + j]*xv;
		}
	}
	y[index] = sum;
*/
}

#include "AtomicAdd.h"
__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K)
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

/*	int index = (blockIdx.x * blockDim.x + threadIdx.x); 	

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
*/
}

extern "C" {
#include <cublas.h>
}

void MatMult_cublas(const double* d_A, const double* d_B, double* d_C, int M, int N, int K)
//
// Transposed matrix-vector multiplication using CUBLAS on GPU
//
{



}
