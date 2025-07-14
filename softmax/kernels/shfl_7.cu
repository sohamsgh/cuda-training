#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
This one is largely similar to the softmax 6 kernel which reduces in shared memory. 
The difference is instead of accessing
shared memory and having sync barrier overhead, we will use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp (group of 32 threads) execute
an instuction parallely on GPU so no chance of race conditions.
*/
__global__ void softmax_kernel_7(float* xd, float* resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[64];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int i = threadIdx.x;
    // we will reduce inside wach warp in the block. 
    // so we need to know which warp in the block and which lane in the 
    // warp we are in.
    unsigned int warpID = tid / warpSize;
    unsigned int lane = tid % warpSize;
    unsigned mask = 0xffffffff; // All threads in the warp are active
    // number of threads in a warp
    //unsigned int warp_size = 32; // not necessary, CUDA provides warpSize
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


    // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
    // note: if there are 256 threads in one block (8 warps of 32 threads each)
    // the following for loop reduces the value in all the 8 warps
    // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
    // float val = smem[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float max1 = local_max;
        float max2 = __shfl_down_sync(mask, max1, offset);
        local_max = fmaxf(max1, max2);
        float norm1 = local_norm;
        float norm2 = __shfl_down_sync(mask, norm1, offset);
	// correct each partial norm using property of exponentials and reduce
	// One of the two reduction terms should be 1
	local_norm = norm1 * expf(max1 - local_max) + 
		         norm2 * expf(max2  - local_max);
    }

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp

    //The if condition probably wont improve performance since
    //without it the compiler will put the (upto 32) shared memory writes in 
    // one instruction which will be just as fast as a single write, 
    // and will also not have to deal with an if condition
    // We are looking at essentially two smem writes versus one smem write and one if.
    if (blockDim.x > warpSize) {  // the case for more than one warp
        // from lane 0 of each warp, write the warp-reduced value to shared memory
        if (lane == 0) {
            // which warp are we at?
            // store the max value in shared memory's warpID index
	    // store the norm value ion shared memory's warpID+warpSize index
            smem[warpID] = local_max;
            smem[warpID + warpSize] = local_norm;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (warpID == 0) {
	    // check if there is meaningful memory corresponding to each lane
	    // otherwise load -INFINITY for max and 0 for norm
            local_max = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid] : -INFINITY;
            local_norm = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid + warpSize] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                float max1 = local_max;
        	float max2 = __shfl_down_sync(mask, max1, offset);
        	local_max = fmaxf(max1, max2);
        	float norm1 = local_norm;
        	float norm2 = __shfl_down_sync(mask, norm1, offset);
		// correct each partial norm using property of exponentials and reduce
		// One of the two reduction terms should be 1
		local_norm = norm1 * expf(max1 - local_max) + 
			         norm2 * expf(max2  - local_max);
            }
            if (tid == 0) {
		    smem[0] = local_max;
		    smem[warpSize] = local_norm;
	    }
        }
    } else {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tid == 0) {
		smem[0] = local_max;
		smem[warpSize] = local_norm;
	}
    }
    __syncthreads();

    float row_max = smem[0];
    float row_norm = smem[warpSize];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 3`
*/
float run_kernel_7(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(512);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_7<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}
