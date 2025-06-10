#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

#define EXP_A (1048576/M_LN2) // 2^20/ln(2) 
#define EXP_C 30801

#define cudaCheck(msg) (cudacheck(msg, __FILE__, __LINE__))


// error handling function
void cudacheck(const char *msg, const char* file, int line) {
        cudaError_t __error = cudaGetLastError();
        if (__error != cudaSuccess) {
                fprintf(stderr, "[FATAL CUDA ERROR] : %s (%s at %s:%d )\n", msg,
                                cudaGetErrorString(__error), file, line);
                fprintf(stderr, "***Failed, Aborting***\n");
                exit(1);
        }
}

__device__ inline 
double ecoexp_original_device(double y) {

	if (y > 700 ) return HUGE_VAL;
	if ((y < -700) return 0.0;

	// Original Schraudolph
	union {
		double d;
		struct {
#ifdef LITTLE_ENDIAN
			int j, i;
#else
			int i, j;
#endif
		} n;
	}_eco;

	_eco.n.i = (int) (EXP_A*(y)) + (1072693248 - EXP_C);
	_eco.n.j = 0;

	return _eco.d;
}


__device__ inline 
double ecoexp_schraudolph_corrected_device(double y) {

	if (y > 700 ) return HUGE_VAL;
	if (y < -700) return 0.0;

	// Original Schraudolph
	union {
		double d;
		struct {
#ifdef LITTLE_ENDIAN
			int j, i;
#else
			int i, j;
#endif
		} n;
	}_eco;

	_eco.n.i = (int) (EXP_A*(y)) + (1072693248 - EXP_C);
	_eco.n.j = 0;

	double base =  _eco.d;

	// correction term 
	double y2 = y*y;
	double correction = 1.0 + y2 * (0.0001 + y2 * 0.000001);
	return base*correction;
}

__device__ inline 
double ecoexp_improved1_device(double y) {

	// range reduction
	double k = floor(y / M_LN2 + 0.5); // floor is a CUDA double function
    	double r = y - k * M_LN2;

	// Pade approximation for e^r
	double r2 = r * r;
        double numerator = 2.0 + r + r2 / 6.0;
        double denominator = 2.0 - r + r2 / 6.0;
        double exp_r = numerator / denominator;

	// CUDA's ldexp alternative
	return exp2(k) * exp_r;
}

__device__ inline 
double ecoexp_improved2_device(double y) {

	// range reduction
	double k = floor(y / M_LN2 + 0.5); // floor is a CUDA double function
    	double r = y - k * M_LN2;

	// 5th order polynomial approximation
        double r2 = r * r;
        double r3 = r2 * r;
        double r4 = r2 * r2;
        double r5 = r4 * r;

        double exp_r = 1.0 + r + r2*0.5 + r3/6.0 + r4/24.0 + r5/120.0;

	// CUDA's ldexp alternative
        return exp2(k) * exp_r;
}

// test the kernels
__global__ 

void test_fast_exp_kernels(double* input, double* output_std, 
		double* output_fast, double* errs, int n, int method) {

	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	double result_fast;

	// grid-strided loop
	for (int i = idx; i < n; i += gridDim.x*blockDim.x) {

		double x = input[i];
		double result_std = exp(x)

		// test the various fast kernels
		switch (method) {
			case 0: result_fast = ecoexp_original_device(x); break;
			case 1: result_fast = ecoexp_schraudolph_corrected_device(x); break;
			case 2: result_fast = ecoexp_improved1_device(x); break;
			case 3: result_fast = ecoexp_improved2_device(x); break;
			default: result_fast = ecoexp_improved2_device(x); break;
		}

		output_std[i] = result_std;
		output_fast[i] = result_fast;
		errs[i] = std::abs(result_std - result_fast);

	}

}


// Peeformance kernel
__global__
void bench_fast_exp_kernels(double* input, double* output, 
		int n, int method, int iteration, double* weight) {

	unsogned int idx = lockIdx.x*blovkDim.x + threadIdx.x;
	double result = 0.0f;

	// grid-strided loop
	for (int i = idx; i< n; i+=gridDim.x*blockDim.x) {

		// iterations
		for (int j = 0; j < iterations; j++)n {
			switch (method) {
				case 0: result += exp(x + weight*j); break;  // standard exponential
				case 1: result += ecoexp_improved2_device(x+weight*j); break;  // fast exponential
				default: result += ecoexp_improved2_device(x+weight*j); break;  // fast exponential
			}
		}
		output[i] = result;
	}
}

			
