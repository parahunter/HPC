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


/*
#include "AtomicAdd.h"
__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K)
{
	//reversed order for better coalescence 
	int tid_x = ( blockIdx.x * blockDim.x + threadIdx.x ) * 4;
	int tid_y = ( blockIdx.y * blockDim.y + threadIdx.y ) * 4;
	
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

//	for(int i = 0 ; i < 4 * 4 ; i++)				
	//	sm[ i] = 0.0;

	for(int k = 0; k < K; k++) 
	{
		for(int i = 0 ; i < 4 ; i++)
		{	
			for(int j = 0 ; j < 4 ; j++)
			{
				sm[j + 4 * i ] += A[k + (tid_y + i) * K]  * B[ k*N + (tid_x + j)];  
			}
		}
	}

	for(int i = 0 ; i < 4 ; i++)	
	{
		for(int j = 0 ; j < 4 ; j++)
		{
			C[(i+tid_y) * N + (j+tid_x)] = sm[j + 4 * i ];
		}
	}
}
*/

#include "AtomicAdd.h"
__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K)
{
	int blk_x = blockIdx.x * blockDim.x * 4;
	int blk_y = blockIdx.y * blockDim.y * 4;

	//printf("%d,%d,%d,%d\n",blockIdx.x, blockDim.x, blockDim.y, gridDim.x);
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


	for(int k = 0; k < K; k++) 
	{
		for(int i = 0 ; i < 4 ; i++)
		{	
			for(int j = 0 ; j < 4 ; j++)
			{
				int ix = blk_x + i * blockDim.x;
				int ixx = ix + threadIdx.x;
				int iy = blk_y + j * blockDim.y;
				int iyy = iy + threadIdx.y;

					sm[j + 4 * i ] += A[k + iyy * K]  * B[ k*N + ixx];  
				}
	}
	}
	
	
	for(int i = 0 ; i < 4 ; i++)	
	{
		for(int j = 0 ; j < 4 ; j++)
		{

			int ix = blk_x + i * blockDim.x;
			int ixx = ix + threadIdx.x;
			int iy = blk_y + j* blockDim.y;
			int iyy = iy + threadIdx.y;
			C[iyy * N + ixx] = sm[j + 4 * i ];
		}
	}
}

__global__ void MatMult_kernel_v3(const double* A, const double* B, double* C, int M, int N, int K)
{
/*

Suggested steps:
1 - Allocate shared memory
2 - Block the k loop - should still work
3 - Every time I add in steps of k - block shared memory
4 - Change one global memory access to oshared memory access
5 - Finally do the same for B
*/

	int blk_x = blockIdx.x * blockDim.x * 4;
	int blk_y = blockIdx.y * blockDim.y * 4;

	//printf("%d,%d,%d,%d\n",blockIdx.x, blockDim.x, blockDim.y, gridDim.x);
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


	__shared__ double A_s[64][16];
	__shared__ double B_s[16][64];	

	int blockK = 16;
	
	for(int l = 0; l < K/blockK; l++) {
		for(int i = 0; i < 4; i++) {
			int ix = blk_x + l * blockDim.x;
			int ixx = ix + threadIdx.x;
			int iy = blk_y + i * blockDim.y;
			int iyy = iy + threadIdx.y;
		
			A_s[iyy - blk_y][threadIdx.x] = A[ixx + iyy * K];

			ix = blk_x + i * blockDim.x;
			ixx = ix + threadIdx.x;
			iy = blk_y + l * blockDim.y;
			iyy = iy + threadIdx.y;

			B_s[threadIdx.y][ixx - blk_x] = B[ixx + iyy * K];

		}

		__syncthreads();
		for(int kk = 0; kk < blockK; kk++) {
			int k = kk + blockK * l;
			for(int i = 0 ; i < 4 ; i++)
			{	
				for(int j = 0 ; j < 4 ; j++)
				{
					int ix = blk_x + i * blockDim.x;
					int ixx = ix + threadIdx.x;
					int iy = blk_y + j * blockDim.y;
					int iyy = iy + threadIdx.y;

						sm[j + 4 * i ] += A_s[iyy - blk_y][kk] * B[ k*N + ixx];//[k + iyy * K]  * B[ k*N + ixx];  
					}
			}
		}
	}
	
	for(int i = 0 ; i < 4 ; i++)	
	{
		for(int j = 0 ; j < 4 ; j++)
		{

			int ix = blk_x + i * blockDim.x;
			int ixx = ix + threadIdx.x;
			int iy = blk_y + j* blockDim.y;
			int iyy = iy + threadIdx.y;
			C[iyy * N + ixx] = sm[j + 4 * i ];
		}
	}

/*
	//reversed order for better coalescence 
	int tid_x = ( blockIdx.x * blockDim.x + threadIdx.x ) * 4;
	int tid_y = ( blockIdx.y * blockDim.y + threadIdx.y ) * 4;

	__shared__ double A_s[64][16];
	__shared__ double B_s[16][64];	

	

	//printf("%d, %d, %d\n", blockDim.x, gridDim.x, threadIdx.x);

	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	int blockK = 16;

	for(int l = 0; l < K/blockK; l++) {

		for(int i = 0 ; i < 4 ; i++) //every thread loads 64*16/256 = 4 elements per A,B
		{	
				A_s[threadIdx.x][threadIdx.y * 4 + i] = A[l * blockK + i + (tid_y) * K];
				B_s[threadIdx.x * 4 + i][threadIdx.y] = B[(l * blockK + i)*N + (tid_x)];	
				
			//		printf("%d, %d\n", threadIdx.x* 4 + i, threadIdx.y);
		}

		__syncthreads();

		for(int kk = 0; kk < blockK ; kk++) 
		{
			int k = kk + blockK * l;
			for(int i = 0 ; i < 4 ; i++)
			{	
				for(int j = 0 ; j < 4 ; j++)
				{
			//		if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) 
						printf("§%f, %f\n", A[k + (tid_y + i) * K], B[ k*N + (tid_x + j)]);  					

					sm[j + 4 * i] += A[k + (tid_y + i) * K]  * B[ k*N + (tid_x + j)];  					
				}
			}
		}

		for(int kk = 0; kk < blockK ; kk++) 
		{
			int k = kk + blockK * l;
			for(int i = 0 ; i < 4 ; i++)
			{	
				for(int j = 0 ; j < 4 ; j++)
				{
			//		if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) 
			//			printf("$%f, %f\n", A_s[i][kk], B_s[kk][j]);  					
				}
			}
		}
		
		//__syncthreads();
	}

	for(int i = 0 ; i < 4 ; i++)	
	{
		for(int j = 0 ; j < 4 ; j++)
		{
			C[(i+tid_y) * N + (j+tid_x)] = sm[j + 4 * i ];
		}
	}
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
	cublasDgemm('N','N', M, N, K, 1.0, d_B, N, d_A, K, 0.0, d_C, N);
}
