#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "Jacobi.h"
#include "possion.h"
#include <omp.h>

int main(int argc, char *argv[])
{

	int N, iterations;
	dim3 threadsPerBlock, blocksPerGrid;

	double *d_A1, *d_A2, *d_B1, *d_B2;
	double *h_src, *h_A0, *h_A1, *h_A2, *h_B0;
	cudaDeviceProp deviceProp;

/***************************
 * Input & info
 ***************************/
	char debug='F';
	if (argc>1 ? N = atoi(argv[1]) : N = 128);
	if (argc>2 ? iterations = atoi(argv[2]) : iterations = 100);
	if (argc>3 ? threadsPerBlock.x = atoi(argv[3]) : threadsPerBlock.x = 16);
	if (argc>4 ? debug = argv[4][0] : debug='F');

	// blocks to cover all M elements in output vector
	threadsPerBlock.y=threadsPerBlock.x;
	blocksPerGrid.x = (N+threadsPerBlock.x-1)/threadsPerBlock.x;
	blocksPerGrid.y=blocksPerGrid.x;
	
	// Check limitation of available device
	cudaGetDeviceProperties(&deviceProp, 0); // assumed only one device = 0
	if(debug=='T')
	{
		printf("Poisson problem.\n");
		printf("  Usage: ./Jacobi <N:default=128> <Iterations:default=100>  <threadsPerBlockLine:default=16> <debugMode [T|F]:default=F>\n\n");

		printf("Device 0: \"%s\".\n", deviceProp.name);
		printf("  Maximum number of threads per block: %d.\n\n", deviceProp.maxThreadsPerBlock);
		if (threadsPerBlock.x*threadsPerBlock.y > deviceProp.maxThreadsPerBlock)
		{
			printf("Error : threadsPerBlock exceeds maximum.\n"); exit(0);
		}
		printf("Threads per block line= %d.\n",threadsPerBlock.x);
		printf("Number of blocks [gridDim.x] = %d.\n",blocksPerGrid.x);
		printf("Matrix size [NxN] = %dx%d.\n\n",N,N); 
	}
/***************************
 * Initialization of memory*
 ***************************/

	int size = N * N * sizeof(double);
	h_src = (double *) calloc(N*N, sizeof(double));
	h_A0 = (double *) malloc(size);
	h_A1 = (double *) malloc(size);
	h_A2 = (double *) malloc(size);
	
	h_B0 = (double *) malloc(size); //cpu

	initMat(&h_src[0], N);
	//init here

/***************************
 * Timers
 ***************************/


	double time_gold;
	double time_v1, transfer_v1;
	double time_v2, transfer_v2;

	StopWatchInterface *timer1, *timer2;
	sdkCreateTimer(&timer1);
	sdkCreateTimer(&timer2);
    cudaDeviceSynchronize();


/***************************
 * CPU gold execution      *
 ***************************/

	sdkResetTimer(&timer1);
	sdkStartTimer(&timer1);

	for(int i=0; i<N*N; i++)
		h_B0[i]=h_A0[i]=h_src[i];

	Jacobi_gold(&h_A0[0], &h_B0[0],iterations, N);
  cudaDeviceSynchronize();
	sdkStopTimer(&timer1);
	time_gold = sdkGetTimerValue(&timer1);


/***************************
 * GPU execution v1        *
 ***************************/
	checkCudaErrors(cudaMalloc((void**)&d_A1, size)); 
	checkCudaErrors(cudaMalloc((void**)&d_B1, size)); 

	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);


	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A1, h_src, size, cudaMemcpyHostToDevice)); 
  cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	// launch kernel
	sdkStartTimer(&timer2);
	
	for(int k=0; k<iterations; k++)
	{
		Jacobi_v1<<<blocksPerGrid, threadsPerBlock, 0>>>(d_A1,d_B1,N);	
//		cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
		Jacobi_v1<<<blocksPerGrid, threadsPerBlock, 0>>>(d_B1,d_A1,N);
		cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	}
	sdkStopTimer(&timer2);

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_A1, d_A1, size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	transfer_v2 = sdkGetTimerValue(&timer1);
	time_v2 = sdkGetTimerValue(&timer2);

 cudaFree(d_A1);
 cudaFree(d_B1);

/***************************
 * GPU execution v2        *
 ***************************/
	checkCudaErrors(cudaMalloc((void**)&d_A2, size)); 
	checkCudaErrors(cudaMalloc((void**)&d_B2, size)); 
	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);


	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A2, h_src, size, cudaMemcpyHostToDevice)); 
  cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	// launch kernel
	sdkStartTimer(&timer2);
	int sharedMem = (2+threadsPerBlock.x)*(2+threadsPerBlock.y)*sizeof(double);
	for(int k=0; k<iterations; k++)
	{
		Jacobi_v2<<<blocksPerGrid, threadsPerBlock, sharedMem>>>(d_A2,d_B2,N);	
		cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
		Jacobi_v2<<<blocksPerGrid, threadsPerBlock, sharedMem>>>(d_B2,d_A2,N);
		cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	}

	sdkStopTimer(&timer2);

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_A2, d_A2, size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	transfer_v1 = sdkGetTimerValue(&timer1);
	time_v1 = sdkGetTimerValue(&timer2);

 cudaFree(d_A2);
 cudaFree(d_B2);
/***************************
 * Verification & timings  *
 ***************************/

	// calculate two-norms of result vectors
	double norm_gold, norm_v1, norm_v2;
	for (int i = 0; i < N*N; ++i)
	{
		norm_gold += h_A0[i]*h_A0[i];
		norm_v1 += h_A1[i]*h_A1[i];
		norm_v2 += h_A2[i]*h_A2[i];
	}
	norm_gold = norm_gold/(N*N);
	norm_v1 = norm_v1/(N*N);
	norm_v2 = norm_v2/(N*N);
	double 	norm=norm_gold;
	double time_blas=time_gold;
	// output verification and timings
	
	if(debug=='T')
	{
    printf("  CPU gold time                 : %3.2f (ms) , speedup %.2fx\n",time_gold,time_blas/time_gold);
    printf("  CPU gold flop                 : %3.2f (Gflops) \n",(double)N*N*iterations*2/time_gold/1e6);
    if (abs(norm-norm_gold)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );
    printf("  GPU v1 time compute           : %3.2f (ms) , speedup %.2fx\n",time_v1,time_blas/time_v1);
	printf("  GPU v1 time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_v1+transfer_v1,time_blas/(time_v1+transfer_v1));
    printf("  GPU v1 flops device           : %2.2f (Gflops) \n",(double)N*N*iterations*2/time_v1/1e6);
    printf("  GPU v1 flops host-device-host : %2.2f (Gflops) \n",(double)N*N*iterations*2/(time_v1+transfer_v1)/1e6);
    if (abs(norm-norm_v1)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED: CPU=%f GPU1=%f\n\n",norm,norm_v1)  );
    printf("  GPU v2 time compute           : %3.2f (ms) , speedup %.2fx\n",time_v2,time_blas/time_v2);
	printf("  GPU v2 time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_v2+transfer_v2,time_blas/(time_v2+transfer_v2));
    printf("  GPU v2 flops device           : %2.2f (Gflops) \n",(double)N*N*iterations*2/time_v2/1e6);
    printf("  GPU v2 flops host-device-host : %2.2f (Gflops) \n",(double)N*N*2/(time_v2+transfer_v2)/1e6);
    if (abs(norm-norm_v2)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );
}else
{ 
		printf("%d \t %d \t %d \t",N,iterations,threadsPerBlock.x);

		printf("%3.2f \t %.2f \t",time_gold,time_blas/time_gold); //cpu time and speedup
printf("%3.2f \t %.2f \t",time_gold,time_blas/time_gold); //cpu time 2 and speedup
    printf("%3.2f \t",(double)N*N*iterations*2/time_gold/1e6); //cpu gflop
    printf("%3.2f \t",(double)N*N*iterations*2/time_gold/1e6); //cpu gflop
        
    printf("%3.2f \t %.2f \t",time_v1,time_blas/time_v1); //gpu 1 time
	printf("%3.2f \t %.2f \t",time_v1+transfer_v1,time_blas/(time_v1+transfer_v1)); // gpu 1 time 2
    printf("%2.2f \t",(double)N*N*iterations*2/time_v1/1e6); //gpu 1 flop 1
    printf("%2.2f \t",(double)N*N*iterations*2/(time_v1+transfer_v1)/1e6); //gpu 1 flops 2
    
    printf("%3.2f \t %.2f \t",time_v2,time_blas/time_v2); //gpu 2 time 1
	printf("%3.2f \t %.2f \t",time_v2+transfer_v2,time_blas/(time_v2+transfer_v2)); //gpu 2 time 2
    printf("%2.2f \t",(double)N*N*iterations*2/time_v2/1e6); //flops 1
    printf("%2.2f\n",(double)N*N*2/(time_v2+transfer_v2)/1e6); //flops 2
    }
/*
print(h_A0, N);
printf("\n\n\n");
print(h_A1, N);

printf("\n\n\n");
print(h_A2, N);*/
/***************************
 * Cleaning memory         *
 ***************************/

 return 0;

}
