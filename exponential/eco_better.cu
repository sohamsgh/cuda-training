#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

#define EXP_A (1048576/M_LN2) // 2^20/ln(2) 
#define EXP_C 30801

#define cudaCheck(msg) (cudacheck(msg, __FILE__, __LINE__))

// data buffer and kernel sizes
const size_t N = 8ULL*1024ULL*1024ULL; // 
const size_t MEM_SIZE = N*sizeof(double); // memory required for input vector
const int BLOCK_SIZE = 256;
const int BLOCK_SIZE_2 = 128;
const double VAL = 0.1;
const double TOLERANCE = 0.00001;
const char* METHOD_NAMES[] = {"Original Schraudolph", "Corrected Schraudolph", "Pade Approximant", "5th Order Polynomial"};

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

int ceil_div(int numerator, int denominator) {
        std::div_t res = std::div(numerator, denominator);
        return res.rem? (res.quot + 1) : res.quot;
}

void init_input(double *h_input, int n, double val) {
	for (int i = 0; i < N; i++){
		h_input[i] = ((double) (i - N/2))/(0.1*N); // keep the domain from -5 to 5
	}
}
	
int find_max_error(double *input, double *errors, int n, double tolerance) {
        int num_failures = 0;
	double max_err = 0.0;
	double max_err_x = 0.0;
	for (int i = 0; i < n; i++) {
	       if (errors[i] > tolerance) {
		       num_failures++;
	       }
	       if (errors[i] > max_err) {
		       max_err = errors[i];
		       max_err_x = input[i];
	       }
	}

	printf("  Max error: %e at x = %8.4f\n", max_error, max_error_x);
    	printf("  Failures (>%.6f): %d/%d\n", tolerance, num_failures, n);

        printf ("Bandwidth Results:\n");
        printf("MEMORY SIZE (MBytes): %12.2f, time in ms: %12.4f, \
        Bandwidth (GB/s): %12.4f\n", float(MEM_SIZE)*1e-06, ms, (2*float(MEM_SIZE)*1e-06 )/ ms);
	printf("------------------------------\n");
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

// Host functions
// Test accuracy of the kernels

void test_accuracy(int method) {

	// buffers
	// host buffers
	double *h_input, *h_output_std, *h_output_fast, *h_errs;
	// device buffers
	double *d_input, *d_output_std, *d_output_fast, *d_errs;
	// allocate host memory
	h_input = new double[N];
	h_output_std = new double[N];
	h_output_fast = new double[N];
	h_errs = new double[N];
	// allocate device memory
	cudaMalloc(&d_input, MEM_SIZE);	
	cudaMalloc(&d_output_std, MEM_SIZE);	
	cudaMalloc(&d_output_fast, MEM_SIZE);
	cudaMalloc(&d_errs, MEM_SIZE);
	cudaCheck("cuaMalloc failure");	
	// intialize input data
	init_input(h_input, N, VAL);
	// copy to device
	cudaMemcpy(d_input, h_input, MEM_SIZE, cudaMemcpyHosttoDevice);

	// timing variable
	float ms;
	// grid and block dimensions
	dim3 dinGrid(ceil_div(N, BLOCK_SIZE));
	dim3dimBlock(BLOCK_SIZE);
    	printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
    	dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

        // Create events for timing
    	cudaEvent_t startEvent, stopEvent;
    	cudaEventCreate(&startEvent);
    	cudaEventCreate(&stopEvent);
    	cudaCheck("cudaeventCreateFailure");
    	// Timings
    	printf("%25s %25s\n", "Routine", "Bandwidth (GB/s)");
	// Test kernel
	printf("%25s\n", "Launching test kernels");
	// record the event
    	cudaEventRecord(startEvent, 0);
    	cudaCheck("cudaEventRecord failure");
	// Print the method being used
	printf("Method %d (%s):\n", method, METHOD_NAME[method]);
    	// launch kernel
    	test_fast_exp_kernels<<<dimGrid, dimBlock>>>(d_input, d_output_std, d_output_fast, d_errs, n, method);
    	 cudaCheck("reduction_1_interleaved kernel launch failure");
    	cudaEventRecord(stopEvent, 0);
    	cudaCheck("reduction_1_interleaved kernel execution failure of cudaEventRecord failure");
    	printf("%25s\n", "Interleaved: warp divergence done");
    	cudaEventSynchronize(stopEvent);
    	cudaCheck("cudaEventSynchronize failure");
    	cudaEventElapsedTime(&ms, startEvent, stopEvent);
    	cudaCheck("cudaEventElapsedTime failure");
	// copy results back
	cudaMemcpy(h_output_std, d_output_std, MEM_SIZE, cudaMemcpyDevicetoHost);
	cudaMemcpy(h_output_fast, d_output_fast, MEM_SIZE, cudaMemcpyDevicetoHost);
	cudaMemcpy(h_errs, d_errs, MEM_SIZE, cudaMemcpyDevicetoHost);
	// Check errors and print timings
	find_max_error(h_input, h_errors, N, TOLERANCE);

	// free host memory
	delete[] h_input;
	delete[] h_output_std;
	delete[] h_output_fast;
	delete[] h_errs;
	// free device memory
	cudaFree(d_input);
	cudaFree(d_output_std);
	cudaFree(d_output_fast;
	cudaFree(d_errs);

}
	
