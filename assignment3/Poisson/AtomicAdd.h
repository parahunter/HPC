#ifndef __ATOMICADD_H__
#define __ATOMICADD_H__

//
// If capability is less than 2.0 output error
//
#if defined (__CUDA_ARCH__) && __CUDA_ARCH__ < 200
#error Cuda capability less than 2.0 is not supported!
#endif

inline __device__ void atomicAdd(double* address, double val) 
{ 
	double old = *address, assumed; 
	do { 
		assumed = old; 
		old = __longlong_as_double( atomicCAS((unsigned long long int*)address, 
											  __double_as_longlong(assumed), 
											  __double_as_longlong(val + assumed))); 
	} while (assumed != old);
}

#endif
