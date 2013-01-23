#ifndef __POISSON_H__
#define __POISSON_H__

void Jacobi_gold(double* u1, double* u2, int iterations, int N);
__global__ void Jacobi_v1(double* u1, double* u2, int N);
__global__ void Jacobi_v2(double* u1, double* u2, int N);
#endif
