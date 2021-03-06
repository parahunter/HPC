#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "MatMult.h"

#include <omp.h>


int main(int argc, char *argv[])
{

	int M, N, K, threadsPerBlock, reps, blocksPerGrid;
	unsigned int size_A, size_B, size_C;
	double *d_A, *d_B, *d_C;
	double *h_A, *h_B, *h_C, *h_C0, *h_C1, *h_C2, *h_C3;
	cudaDeviceProp deviceProp;
	char mode;

/***************************
 * Input & info
 ***************************/

	printf("Matrix-vector multiplication.\n");
	printf("  Usage: ./MatMult <M:default=4096> <N:default=4096> <K:default=4096>  <threadsPerBlock:default=128> <reps:default=10>\n\n");
	if (argc>1 ? M = atoi(argv[1]) : M = 4096);
	if (argc>2 ? N = atoi(argv[2]) : N = 4096);
	if (argc>3 ? K = atoi(argv[3]) : K = 4096);
	if (argc>4 ? threadsPerBlock = atoi(argv[4]) : threadsPerBlock = 16);
	if (argc>5 ? reps = atoi(argv[5]) : reps = 10);
	if (argc>6 ? mode = argv[6][0] : mode = 'd');


	// blocks to cover all M elements in output vector
	blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

	// Check limitation of available device
	cudaGetDeviceProperties(&deviceProp, 0); // assumed only one device = 0
	printf("Device 0: \"%s\".\n", deviceProp.name);
	printf("  Maximum number of threads per block: %d.\n\n", deviceProp.maxThreadsPerBlock);
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
	{
		printf("Error : threadsPerBlock exceeds maximum.\n"); exit(0);
	}
	printf("Threads per block = %d.\n",threadsPerBlock);
	printf("Number of blocks [gridDim.x] = %d.\n",blocksPerGrid);
	printf("Matrix size [MxN] = %dx%d.\n\n",M,N); 
		
/***************************
 * Initialization of memory*
 ***************************/

	size_A = M * K * sizeof(double);
	size_B = K * N * sizeof(double);
	size_C = M * N * sizeof(double);
	h_A = (double *) malloc(size_A);
	h_B = (double *) malloc(size_B);
	h_C = (double *) malloc(size_C);  // reference blas call output
	h_C0 = (double *) malloc(size_C); // gold output
	h_C1 = (double *) malloc(size_C); // kernel v1 output
	h_C2 = (double *) malloc(size_C); // kernel v2 output
	h_C3 = (double *) malloc(size_C); // cublas output
	for (int i = 0; i < M * K; ++i) {
		h_A[i] = rand()/(double)RAND_MAX;
	}
	for (int i = 0; i < K * N; ++i) {
		h_B[i] = rand()/(double)RAND_MAX;
	}
	checkCudaErrors(cudaMalloc((void**)&d_A, size_A)); 
	checkCudaErrors(cudaMalloc((void**)&d_B, size_B)); 
	checkCudaErrors(cudaMalloc((void**)&d_C, size_C)); 

/***************************
 * Timers
 ***************************/

	double time_blas;
	double time_v3, transfer_v3;
	double time_v1, transfer_v1;
	double time_v2, transfer_v2;
	double time_cublas, transfer_cublas;
	StopWatchInterface *timer1, *timer2;
	sdkCreateTimer(&timer1);
	sdkCreateTimer(&timer2);
    cudaDeviceSynchronize();

/***************************
 * CPU BLAS execution           *
 ***************************/

	sdkResetTimer(&timer1);
	sdkStartTimer(&timer1);
	for (int iter = 0; iter < reps; ++iter) 
	{
		MatMult_blas(h_A, h_B, h_C, M, N, K);
	}
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);
	time_blas = sdkGetTimerValue(&timer1)/reps;


/***************************
 * GPU execution v1        *
 ***************************/

	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);
	checkCudaErrors(cudaMemset(d_C, 0, size_C)); 

	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	dim3 blockSize(threadsPerBlock, threadsPerBlock);
			
	int blocksPerGridN = (N+threadsPerBlock-1)/threadsPerBlock;
	int blocksPerGridM = (M+threadsPerBlock-1)/threadsPerBlock;
	
	dim3 gridSize(blocksPerGridN, blocksPerGridM);
	
	// launch kernel
	sdkStartTimer(&timer2);
	printf("N: %d  M: %d  K: %d  ",N, M, K);
	for (int iter = 0; iter < reps; ++iter) 
	{
		MatMult_kernel_v1<<<gridSize, blockSize, 0>>>(d_A,d_B,d_C,M,N,K);
	}
    cudaDeviceSynchronize();
	sdkStopTimer(&timer2);

	// check for launch failure
	checkCudaErrors(cudaGetLastError());

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_C1, d_C, size_C, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	transfer_v1 = sdkGetTimerValue(&timer1);
	time_v1 = sdkGetTimerValue(&timer2)/reps;

/***************************
 * GPU execution v2        *
 ***************************/

	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);
	checkCudaErrors(cudaMemset(d_C, 0, size_C)); 

	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	blockSize = dim3(threadsPerBlock, threadsPerBlock);
			
	blocksPerGridN = (N+threadsPerBlock-1)/threadsPerBlock;
	blocksPerGridM = (M+threadsPerBlock-1)/threadsPerBlock;
	
	gridSize = dim3((blocksPerGridN+3)/4, (blocksPerGridM+3)/4);
	printf("Real grid size (x): %d \n",gridSize.x);
	// launch kernel
	sdkStartTimer(&timer2);
	for (int iter = 0; iter < reps; ++iter) 
	{ 
		checkCudaErrors(cudaMemset(d_C, 0, size_C)); 
 // YOUR TASKS:
 // - Invoke MatVec_kernel_v2 using a 2D grid.
 // Insert code below this line:

		MatMult_kernel_v2<<<gridSize, blockSize>>>(d_A,d_B,d_C,M,N,K);
	}


	cudaDeviceSynchronize();
	sdkStopTimer(&timer2);

	// check for launch failure
	checkCudaErrors(cudaGetLastError());

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_C2, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);


	transfer_v2 = sdkGetTimerValue(&timer1);
	time_v2 = sdkGetTimerValue(&timer2)/reps;

/***************************
 * GPU execution v3      *
 ***************************/

	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);
	checkCudaErrors(cudaMemset(d_C, 0, size_C)); 

	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	blockSize = dim3(threadsPerBlock , threadsPerBlock);
			
	blocksPerGridN = (N+threadsPerBlock-1)/threadsPerBlock;
	blocksPerGridM = (M+threadsPerBlock-1)/threadsPerBlock;
	
	gridSize = dim3((blocksPerGridN+3)/4, (blocksPerGridM+3)/4);
	// launch kernel
	sdkStartTimer(&timer2);
	for (int iter = 0; iter < reps; ++iter) 
	{ 
		checkCudaErrors(cudaMemset(d_C, 0, size_C)); 
		MatMult_kernel_v3<<<gridSize, blockSize>>>(d_A,d_B,d_C,M,N,K);
	}

	
	cudaDeviceSynchronize();
	sdkStopTimer(&timer2);

	// check for launch failure
	checkCudaErrors(cudaGetLastError());

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_C0, d_C, size_C, cudaMemcpyDeviceToHost));
	    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	transfer_v3 = sdkGetTimerValue(&timer1);
	time_v3 = sdkGetTimerValue(&timer2)/reps;
/*
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			printf("%4.2f ", h_C0[i*N+j]);
		}
		printf("\n");
	}*/

/***************************
 * GPU execution cublas        *
 ***************************/

	// reset GPU timers and result vector
	sdkResetTimer(&timer1);
	sdkResetTimer(&timer2);
	checkCudaErrors(cudaMemset(d_C, 0, size_C)); 

	// transfer data to device
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice)); 
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	// launch kernel
	sdkStartTimer(&timer2);
	for (int iter = 0; iter < reps; ++iter) 
	{
		MatMult_cublas(d_A, d_B, d_C, M, N, K);
	}
    cudaDeviceSynchronize();
	sdkStopTimer(&timer2);
	


	// check for launch failure
	checkCudaErrors(cudaGetLastError());

	// transfer result to host
	sdkStartTimer(&timer1);
	checkCudaErrors(cudaMemcpy(h_C3, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
	sdkStopTimer(&timer1);

	transfer_cublas = sdkGetTimerValue(&timer1);
	time_cublas = sdkGetTimerValue(&timer2)/reps;

/***************************
 * Verification & timings  *
 ***************************/

	// calculate two-norms of result vectors
	double norm, norm_v3, norm_v1, norm_v2, norm_cublas;
	for (int i = 0; i < M*N; ++i)
	{
		norm += h_C[i]*h_C[i];
		norm_v3 += h_C0[i]*h_C0[i];
		norm_v1 += h_C1[i]*h_C1[i];
		norm_v2 += h_C2[i]*h_C2[i];
		norm_cublas += h_C3[i]*h_C3[i];
	}
	norm = sqrt(norm);
	norm_v3 = sqrt(norm_v3);
	norm_v1 = sqrt(norm_v1);
	norm_v2 = sqrt(norm_v2);
	norm_cublas = sqrt(norm_cublas);
	
	printf("norm %f norm_v1 %f \n", norm, norm_v3);	
	

	double flops = (double)M*N*K*2;
	// output verification and timings

	if(mode == 'd')
	{
	
    printf("  CPU blas time                 : %3.2f (ms)\n",time_blas);
    printf("  CPU blas flop                 : %3.2f (Gflops) \n\n",flops/time_blas/1e6);
    printf("  GPU v1 time compute           : %3.2f (ms) , speedup %.2fx\n",time_v1,time_blas/time_v1);
	printf("  GPU v1 time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_v1+transfer_v1,time_blas/(time_v1+transfer_v1));
    printf("  GPU v1 flops device           : %2.2f (Gflops) \n",flops/time_v1/1e6);
    printf("  GPU v1 flops host-device-host : %2.2f (Gflops) \n",flops/(time_v1+transfer_v1)/1e6);
    if (abs(norm-norm_v1)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );
    printf("  GPU v2 time compute           : %3.2f (ms) , speedup %.2fx\n",time_v2,time_blas/time_v2);
	printf("  GPU v2 time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_v2+transfer_v2,time_blas/(time_v2+transfer_v2));
    printf("  GPU v2 flops device           : %2.2f (Gflops) \n",flops/time_v2/1e6);
    printf("  GPU v2 flops host-device-host : %2.2f (Gflops) \n",flops/(time_v2+transfer_v2)/1e6);
    if (abs(norm-norm_v2)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );

    printf("  GPU v3 time compute           : %3.2f (ms) , speedup %.2fx\n",time_v3,time_blas/time_v3);
	printf("  GPU v3 time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_v3+transfer_v3,time_blas/(time_v3+transfer_v3));
    printf("  GPU v3 flops device           : %2.2f (Gflops) \n",flops/time_v3/1e6);
    printf("  GPU v3 flops host-device-host : %2.2f (Gflops) \n",flops/(time_v3+transfer_v3)/1e6);
    if (abs(norm-norm_v3)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );

    printf("  GPU cublas time compute           : %3.2f (ms) , speedup %.2fx\n",time_cublas,time_blas/time_cublas);
	printf("  GPU cublas time comp+trans        : %3.2f (ms) , speedup %.2fx\n",time_cublas+transfer_cublas,time_blas/(time_cublas+transfer_cublas));
    printf("  GPU cublas flops device           : %2.2f (Gflops) \n",flops/time_cublas/1e6);
    printf("  GPU cublas flops host-device-host : %2.2f (Gflops) \n",flops/(time_cublas+transfer_cublas)/1e6);
    if (abs(norm-norm_cublas)/norm < 1e-12 ? printf("  PASSED\n\n") : printf("  FAILED \n\n")  );

	}
	else
	{
		printf("%d \t %f \t %f \t %f \t %f \t %f", N, flops/time_blas/1e6, flops/time_v1/1e6 ,flops/time_v2/1e6, flops/time_v3/1e6, flops/time_cublas/1e6);
	}

/***************************
 * Cleaning memory         *
 ***************************/

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);

 return 0;

}
