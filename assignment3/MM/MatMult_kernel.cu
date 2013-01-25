extern "C" {
	#include <cublas.h>
	#include <cblas.h>
}
#include "stdio.h"

void MatMult_blas(const double* A, const double* B, double* C, int M, int N, int K)
// Matrix multiplication using BLAS on CPU
{
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}

__global__ void MatMult_kernel_v1(const double* A, const double* B, double* C, int M, int N, int K)
// Naive version where only global memory and automatic variables are accessed.
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;	
	int index = i * N + j;
	if(i>=M || j>=N) return;
	double sum = 0.0;
	for(int k = 0; k < K; k++) 
	{
		sum += A[k + i * K] * B[ k*N + j];  
	}

	C[index] = sum;
}


/* OLD VERSION - wrong accessing order.

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

__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K)
{
	int ix = (blockIdx.x * blockDim.x * 4) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y * 4) + threadIdx.y;
	
	//printf("%d,%d,%d,%d\n",blockIdx.x, blockDim.x, blockDim.y, gridDim.x);
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	//calculating the 4x4 block into registers
	for(int k = 0; k < K; k++) 
	{
		for(int i = 0 ; i < 4 ; i++)
		{	
			for(int j = 0 ; j < 4 ; j++)
			{
				int ixx = ix + i * blockDim.x;
				int iyy = iy + j * blockDim.y;
				if(ixx<N && iyy<M)
				sm[j + 4 * i ] += A[k + iyy * K] * B[ k*N + ixx];  
			}
		}
	}
	
	//transfering the registers to global memory
	for(int i = 0 ; i < 4 ; i++)	
	{
		for(int j = 0 ; j < 4 ; j++)
		{
			int ixx = ix + i * blockDim.x;
			int iyy = iy + j * blockDim.y;
			if(ixx<N && iyy<M)
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
	int ix = threadIdx.x;
	int iy = threadIdx.y;

	//printf("%d,%d,%d,%d\n",blockIdx.x, blockDim.x, blockDim.y, gridDim.x);
	double sm[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	// allocating shared memory
	__shared__ double A_s[64][16];
	__shared__ double B_s[16][64];	

	//blocking the k loop
	int blockK = 16;

	for(int l = 0; l < K; l += blockK)
	{		
		// filling shared memory
		for(int b = 0; b < 4; b++)
		{
			int off = b*16;
			if(ix + l<K && (blk_y + iy + off)<M)
				A_s[iy + off][ix] = A[(blk_y + iy + off) * K + ix + l];
			else
				A_s[iy + off][ix] = 0;
			if(ix + off + blk_x<N && (iy + l)<K)
				B_s[iy][ix + off] = B[(iy + l) * N + ix + off + blk_x];
			else
				B_s[iy][ix + off]=0;
		}

		// synchronization to ensure the shared memory is filled
		__syncthreads();

		//calculating the values
		for(int k = 0; k < blockK; k++) 
		{
			for(int bb = 0 ; bb < 4 ; bb++)
			{
				for(int ba = 0 ; ba < 4 ; ba++)
				{
					int iya = threadIdx.y + (ba * 16);
					int ixb = threadIdx.x + (bb * 16);
					sm[ba * 4 + bb] += A_s[iya][k] * B_s[k][ixb];
				}
			}
		}
	}
	
	// storing into global memory
	for(int ba = 0 ; ba < 4 ; ba++)
	{	
		for(int bb = 0; bb < 4 ; bb++)
		{
			int ixx = blk_x + bb * blockDim.x + threadIdx.x;
			int iyy = blk_y + ba * blockDim.y + threadIdx.y;
			if(ixx<N && iyy<M)
			C[iyy * N + ixx] = sm[ba * 4 + bb];
		}
	}
}


void MatMult_cublas(const double* d_A, const double* d_B, double* d_C, int M, int N, int K)
//
// Transposed matrix-vector multiplication using CUBLAS on GPU
//
{
	cublasDgemm('N','N', N, M, K, 1.0, d_B, N, d_A, K, 0.0, d_C, N);
}
