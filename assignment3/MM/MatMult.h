#ifndef __MATMULT_H__
#define __MATMULT_H__

void MatMult_gold(const double* A, const double* B, double* C, int M, int N, int K);
void MatMult_blas(const double* A, const double* B, double* C, int M, int N, int K);
__global__ void MatMult_kernel_v1(const double* A, const double* B, double* C, int M, int N, int K);
__global__ void MatMult_kernel_v2(const double* A, const double* B, double* C, int M, int N, int K);
__global__ void MatMult_kernel_v3(const double* A, const double* B, double* C, int M, int N, int K);
void MatMult_cublas(const double* A, const double* B, double* C, int M, int N, int K);

#endif
