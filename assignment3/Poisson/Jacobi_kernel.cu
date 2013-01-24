//
// kernel routine
//
#include "AtomicAdd.h"
#include "stdio.h"

int cc(double a, int n)
{
	n-=2;
	return 1+(int)( 0.5 * (double)n + 0.5 * a * (double) n );
}

__device__ int hcc(double a, int n)
{
	n-=2;
	return 1+(int)( 0.5 * (double)n + 0.5 * a * (double) n );
}

double ff(int i, int j, int n)
{


	if(cc(0,n) < i && i <= cc(1/3.0,n)  && cc(-2/3.0,n) < j && j <= cc(-1-3.0,n) )
		return 200;
	else
		return 0;

}

__device__ double hff(int i, int j, int n)
{


	if(hcc(0,n) < i && i <= hcc(1/3.0,n)  && hcc(-2/3.0,n) < j && j <= hcc(-1-3.0,n) )
		return 200;
	else
		return 0;

}


void updateMat(double* from, double* to, int N)
{
	double hh = (2.0/(N-2))*(2.0/(N-2));
	#pragma omp for schedule(runtime)
	for(int i = 1 ; i < N -1; i++)
	{
		for(int j = 1 ; j < N -1; j++)
		{
			double step = (from[i*N + j-1] + from[(i-1)*N + j] + from[i*N+j+1] + from[(i+1)*N + j] +  hh * ff(i,j,N) )*0.25;
			to[i*N + j] = step;
		}
	}
}


void Jacobi_gold(double* u1, double* u2, int iterations, int N)
{
		#pragma omp parallel
		{
			for(int i=0; i<iterations; i++)
			{
				updateMat(u1, u2, N);
				updateMat(u2, u1, N);
			}
		}
}

__global__ void Jacobi_v1(double* u1, double* u2, int N)
//
// Naive version where only global memory and automatic variables are accessed.
//
{
			double hh = (2.0/(N-2))*(2.0/(N-2));
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;
			if(i>=N || j>=N) return;
			if(i<1 || j<1 || i==N-1 || j==N-1)
			{
				u2[i*N + j]=u1[i*N + j];
			}else
			{
			u2[i*N + j] = (
				u1[i*N + j-1] + u1[(i-1)*N + j] + 
				u1[i*N+j+1] + u1[(i+1)*N + j] +  
				hh * hff(i,j,N) )*0.25;
			}
}

#include "AtomicAdd.h"
__global__ void Jacobi_v2(double* u1, double* u2,int N)
//
// Shared memory
//
{
//	int tid = threadIdx.x;
//	int blkidx = blockIdx.x * gridDim.x + blockIdx.y;
//	int index = (blkidx * blockDim.x + tid); 
			int i = blockIdx.x * blockDim.x + threadIdx.x;
			int j = blockIdx.y * blockDim.y + threadIdx.y;

			double hh = (2.0/(N-2))*(2.0/(N-2));
			if(i>=N || j>=N) return;

			extern __shared__ double data[];
			
			int id = 1+threadIdx.x;
			int nd = (blockDim.x+2);
			int jd = threadIdx.y+1;
			
			data[id*nd+jd]=u1[i*N+j];	
				
			__syncthreads();
			if(i==0|| j==0||i==N-1||j==N-1)
			{
				u2[i*N + j]=data[id*nd + jd];
			}
			else
			{
				if(threadIdx.x==0)
					data[(id-1)*nd+jd]=u1[(i-1)*N+j];
				if(threadIdx.y==0)
					data[id*nd+jd-1]=u1[i*N+j-1];		
				if(threadIdx.x==blockDim.x-1)
					data[(id+1)*nd+jd]=u1[(i+1)*N+j];
				if(threadIdx.y==blockDim.y-1)
					data[id*nd+jd+1]=u1[i*N+j+1];
				u2[i*N + j] = (
					data[id*nd + jd-1] + data[(id-1)*nd + jd] + 
					data[id*nd+jd+1] + data[(id+1)*nd + jd] +  
					hh * hff(i,j,N) )*0.25;
			}
}

