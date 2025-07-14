#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
The previous sharedmem kernel had an issue of half the threads beinfg idle 
on first loop iteration in block-level reduction. 
Solution: Halve the niumber of threads in a block. Replace single loads that compute local
norms and maxes with two loads and do the first level of max and norms there
Before using shared memory.
 
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
In this, we handle each row with a block where the threads within one block work together
to process one row (max and norm factor). Each thread will process some elements
and will contains its local max and local norm in shared memory. Then, we perform reduction
operations to compute the final max and norm factor. Also, we compute maxes and norms
in one pass itself.
*/
__global__ void softmax_kernel_6(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];  // needs to hold local_max and local_norm together , so 2 *(number of threads in a block) floats, 

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int i = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // compute local max and norm for each thread
    // and then finally have a sync barrier before moving on
    // Load two floats from global memory at a time to increase bandwidth
    while (i < N) {
        float x1 = input_row[i];
        float x2 = input_row[i+blockDim.x];
	float x = fmaxf(x1, x2);
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x1 - local_max) + expf(x2 - local_max);
	i += 2*blockDim.x;
    }
    __syncthreads();

    // each thread will have its own local max and local max
    // we store local max in the tid of the shared memory
    // and the local norm in the second half of shared memory
    smem[tid] = local_max;
    smem[tid+blockDim.x] = local_norm;
    __syncthreads();

    // block-level reduction in O(log(N)) time over all threads
    // is faster than linear reduction over all threads
    // We reduce max and norm in the same iteration - increases arithmatic intensity
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
	    float max1 = smem[tid];
	    float max2 = smem[tid+stride];
	    local_max = fmaxf(max1, max2);
	    // correct each partial norm using property of exponentials and reduce
	    // One of the two reduction terms should be 1
	    local_norm = smem[tid + blockDim.x] * expf(max1 - local_max) + 
		         smem[tid + blockDim.x + stride] * expf(max2  - local_max);
	    smem[tid] = local_max;
	    smem[tid + blockDim.x] = local_norm; //note that the operation is '=', not '+='. Reduction happened in local_norm above
        }
        // sync barrier before next iteration to ensure correctness
        __syncthreads();
    }

    // the first element after max reduction from all threads
    // will contain the global max for the row
    // The blokDim.x-th element will contain the global norm for the row
    float row_max = smem[0];
    float row_norm = smem[blockDim.x];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 6`
*/
float run_kernel_6(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(512);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_6<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}
