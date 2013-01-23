#ifndef __MATMULT_H__
#define __MATMULT_H__

void MatMult_gold(const double* A, const double* x, double* y, int M, int N);
void MatMult_blas(const double* A, const double* x, double* y, int M, int N);
__global__ void MatMult_kernel_v1(const double* A, const double* x, double* y, int M, int N);
__global__ void MatMult_kernel_v2(const double* A, const double* x, double* y, int M, int N);
__global__ void MatMult_kernel_v3(const double* A, const double* x, double* y, int M, int N);
void MatMult_cublas(const double* A, const double* x, double* y, int M, int N);

#endif
