#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>


//const size_t N  = 8ULL*1024ULL*1024ULL; // Data size
const size_t N = 8ULL*1024ULL*1024ULL; // data size
const size_t MEM_SIZE = N*sizeof(float); // memory required for input vector
const int BLOCK_SIZE = 256;
const int BLOCK_SIZE_2 = 128;
const float VAL = 1.0f;


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

/*
   Interleaved parallel reduction of floats

   Using the remainder criterion

   Suboptimal solution because it leads to warp divergence

*/

__global__
void reduction_1_interleaved( size_t N, const float* g_idata, float* g_odata)
{
        // declare shared memory
        __shared__ float sdata[BLOCK_SIZE]; // extern allows dynamic shared memory, implicitly determined from 3rd kernel launch param
        unsigned int lidx = threadIdx.x;                                // local thread id
        unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x;      // global thread id
        // each thread loads one element from global memory to shared memory
        if (gidx < N) {
                sdata[lidx] = g_idata[gidx];
        }
        __syncthreads(); // to ensure every thread has loaded before reduction
        //cudaDeviceSynchronize();
        // reduce in shared memory
        for (unsigned int s = 1; s < blockDim.x; s *= 2){
                // s is the stride between the two elements to be reduced
                if (lidx % (2*s) == 0 ){
                        // s, the stride, csn be used to decide which threads hold the redution at each stage
                        //sdata[lidx] += sdata[lidx + s];
                        sdata[lidx] += sdata[lidx + s];
                }
                __syncthreads();
        }
        // synch threads once more is not needed
        // because the last loop after the last syncthreads operats on a single thread - good to ask in an interview
        // Do atomic add so you dont have to launch a (potentially small) reduction kernel afterwards
        if (lidx == 0){
                atomicAdd(g_odata, sdata[0]);
        }

}

__global__
void reduction_2_interleaved(size_t N, const float* g_idata, float* g_odata) {

        //declare shared memory
        __shared__ float sdata[BLOCK_SIZE];
        unsigned int lidx = threadIdx.x;
        unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x;

        // each thread loads one element from global memory to shared memory
        if (gidx < N) {
                sdata[lidx] = g_idata[gidx];
        }
        __syncthreads(); //to ensure every thread has loaded before any thread starts reduction

        // reduce in shared memory

        for (unsigned int s = 1; s < blockDim.x; s *= 2) {
                unsigned int index = 2*s*lidx;

                if (index < blockDim.x) {
                        sdata[index] += sdata[index + s];
                }
                __syncthreads();
        }
        if (lidx == 0 ) {
                atomicAdd(g_odata, sdata[0]);
        }
}
__global__
void reduction_3_sequential(size_t N, const float* g_idata, float *g_odata) {

        // declare shared memory
        __shared__ float sdata[BLOCK_SIZE];
        unsigned int lidx = threadIdx.x;
        unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x;

        // each thread, like before, loads one element from memory
        if (gidx < N) {
                sdata[lidx] = g_idata[gidx];
        }
        __syncthreads();
        // reduce in requestial order in shared memort
        for (unsigned int s = blockDim.x /2; s >0; s>>=1) {
                if (lidx < s) {
                        sdata[lidx] += sdata[lidx + s];
                }
                __syncthreads();
        }
        if (lidx == 0 ) {
                atomicAdd(g_odata, sdata[0]);
        }
}

__global__
void reduction_4_sequential(size_t N, const float* g_idata, float *g_odata) {

        // declare shared memory
        __shared__ float sdata[BLOCK_SIZE_2];
        unsigned int lidx = threadIdx.x;
        unsigned int gidx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        // each thread, like before, loads one element from memory
        if ((gidx + blockDim.x) < N) {
                sdata[lidx] = g_idata[gidx] + g_idata[gidx + blockDim.x];
        }
        __syncthreads();
        // reduce in sequential order in shared memort
        for (unsigned int s = blockDim.x /2; s >0; s>>=1) {
                if (lidx < s) {
                        sdata[lidx] += sdata[lidx + s];
                }
                __syncthreads();
        }
        if (lidx == 0 ) {
                atomicAdd(g_odata, sdata[0]);
        }
}

__device__ 
void warpReduce(volatile float* sdata, unsigned int lidx) {
    
            if (BLOCK_SIZE_2 >= 64) sdata[lidx] += sdata[lidx + 32];
            //__syncthreads();
            if (BLOCK_SIZE_2 >= 32) sdata[lidx] += sdata[lidx + 16];
            //__syncthreads();
            if (BLOCK_SIZE_2 >= 16) sdata[lidx] += sdata[lidx + 8];
            //__syncthreads();
            if (BLOCK_SIZE_2 >= 8) sdata[lidx] += sdata[lidx + 4];
            //__syncthreads();
            if (BLOCK_SIZE_2 >= 4) sdata[lidx] += sdata[lidx + 2];
            //__syncthreads();
            if (BLOCK_SIZE_2 >= 2) sdata[lidx] += sdata[lidx + 1];
            
}

__global__
void reduction_5_sequential(size_t N, const float* g_idata, float *g_odata) {

        
        // declare shared memory
        __shared__ float sdata[BLOCK_SIZE_2];
        unsigned int lidx = threadIdx.x;
        unsigned int gidx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        // each thread, like before, loads one element from memory
        if ((gidx + blockDim.x) < N) {
                sdata[lidx] = g_idata[gidx] + g_idata[gidx + blockDim.x];
        }
        __syncthreads();
        // reduce in requestial order in shared memort
        for (unsigned int s = blockDim.x /2; s >32; s>>=1) {
                if (lidx < s) {
                        sdata[lidx] += sdata[lidx + s];
                }
                __syncthreads();
        }
       
        if (lidx < 32) { 
            warpReduce(sdata, lidx);
        }
        if (lidx == 0 ) {
                atomicAdd(g_odata, sdata[0]);
        }
}

__global__
void reduction_6_sequential(size_t N, const float* g_idata, float *g_odata) {

        // we will reduce as we load as many times as necessary.
        // The relationship between gridblock and data is as follows.
        // Inside a single block we access memory the same way as earlier. 
        // Each thread in a block rduces two input floats and loads in in shared mem.
        // At i and at i+blockDim.x.
        // So a block works on memory 2*blockDim.x. 
        // The previous kernel had 2*blockDim.x*gridDim.x = N (number of input elements). 
        // This kernel has a gridDim.x that is a fraction of earlier.   
        // we go over the input memory in a grid sided loop
        // Inside the loop, for each block we access 
        // declare shared memory
        __shared__ float sdata[BLOCK_SIZE_2];
        unsigned int lidx = threadIdx.x;
        unsigned int gidx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        unsigned int gridSize = gridDim.x * blockDim.x * 2;
        sdata[lidx] = 0.0f;
        // each thread, like before, loads one element from memory
        while (gidx < N) {
                sdata[lidx] +=  g_idata[gidx] + g_idata[gidx + blockDim.x];
                gidx += gridSize;
        }
        __syncthreads();
        // reduce in requestial order in shared memort
        for (unsigned int s = blockDim.x /2; s >32; s>>=1) {
                if (lidx < s) {
                        sdata[lidx] += sdata[lidx + s];
                }
                __syncthreads();
        }
       
        if (lidx < 32) { 
            warpReduce(sdata, lidx);
        }
        if (lidx == 0 ) {
                atomicAdd(g_odata, sdata[0]);
        }
}

__global__ 
void ws_reduce(size_t N, float *g_idata, float *g_odata) {

    // we declare a shared memory blovk the size of one warp
    __shared__ float sdata[128/32];  // number of warps
    unsigned int lidx = threadIdx.x;
    unsigned int gidx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = gridDim.x * blockDim.x * 2;
    float val = 0.0f;
    unsigned  FULL_MASK = 0xFFFFFFFFU;
    // we will reduce inside wach warp in the block. 
    // so we need to know which warp in the block and which lane in the 
    // warp we are in.
    unsigned int warpID = lidx / warpSize;
    unsigned int lane = lidx % warpSize;

    // load into local registers, not shared memory
    while ((gidx + blockDim.x) < N) {
        val += g_idata[gidx] + g_idata[gidx+ blockDim.x];
        gidx += gridSize;
    }   

    // at this point we have one value of val per thread. 
    // first warp shuffle will reduce those values so we 
    // have one value for each warp
    // reduce inside each warp in a tree-like fashion 
    // each step is a sequential reduction
    for (int offset = warpSize / 2; offset > 0; offset>>=1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    } 
    
    // from lane 0 of each warp, write the warp-reduced value to shared memory
    if (lane ==0) {
        sdata[warpID] = val;
    }
    __syncthreads();

    // Hereafter we use only warp 0
    if (warpID ==0) {
        // check that there is meaningful memory corresponding to each lane.
        // if not load 0
        val = (lidx < blockDim.x / warpSize)? sdata[lane] : 0.0f;
        __syncwarp();
        // final warp shuffle
        for (int offset = warpSize / 2; offset > 0; offset>>=1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
    }
  
    if (lidx ==0) atomicAdd(g_odata, val);
}



void init_const_matrix(float *mat, size_t N, const float val) {
        for (int i = 0; i < N; i++) {
                mat[i] = val;
        }
}
void validate_const_matrix(float *mat, unsigned int N, const float val) {
    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (mat[i] != val) {
            printf("%s\n", "*** FAILED ***");
            printf("%s\n", " Constant matrix validation failed at:");
            printf("index: %d, Reference: %f, result: %f", i, val, mat[i]);
            passed = false;
            break;
        }
    }
        if (passed == true) {
            printf("Constant matrix valiodation passed\n");
        }
}
int ceil_div(int numerator, int denominator) {
        std::div_t res = std::div(numerator, denominator);
        return res.rem? (res.quot + 1) : res.quot;
}
void postprocess(const float ref, const float *res, float ms) {
        bool passed = true;
        if (*res != ref) {
                printf("%25s\n", "*** FAILED ***");
                printf("reference: %f result: %f\n", ref, *res);
                passed = false;
        }
        if (passed == true) {
            printf ("Postprocess passed\n");
            printf("MEMORY SIZE (MBytes): %12.2f, time in ms: %12.4f, \
            Bandwidth (GB/s): %12.4f\n", float(MEM_SIZE)*1e-06, ms, (2*float(MEM_SIZE)*1e-06 )/ ms);
	    printf("------------------------------\n");
        }
}

int main(int argc, char *argv[]) {

    int deviceId = 0;
    if (argc > 1) deviceId = atoi(argv[1]);
    
        // buffers
    float *h_A, *h_sum, *d_A, *d_sum;
    h_A = new float[N]; // allocate space in host memory for input matrix
    h_sum = new float;      // new pointer initialized with undeterminate value
    // reference result
    const int ref = N;
    // timing variable
    float ms;
    // grid and block dimensions
    dim3 dimGrid(ceil_div(N, BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE);
    // print device properties
    //cudaDeviceInfo();
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
    dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    // initialize vector with a constant float
    init_const_matrix(h_A, N, VAL);
    validate_const_matrix(h_A, N, VAL);
    cudaMalloc(&d_A, MEM_SIZE);     // allocate memory for the matrix on  device
    cudaMalloc(&d_sum, sizeof(float));      // allocate memory for sum on device
    cudaCheck("cudaMalloc failure");        // cudaCheck uses chudaGetLastError
    // copy matrix A to device
    cudaMemcpy(d_A, h_A, MEM_SIZE, cudaMemcpyHostToDevice);
    cudaCheck("cuda MemCpy H2D failure");
    cudaMemset(d_sum, 0.0f, sizeof(float)); // set result of reduction to zero on device
    cudaCheck("cudaMemset failure");

    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaCheck("cudaeventCreateFailure");
    // Time kernels
    printf("%25s %25s\n", "Routine", "Bandwidth (GB/s)");

    // Interleaved reduction kernel 1
    printf("%25s\n", "Interleaved: warp divergence");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_1_interleaved<<<dimGrid.x, dimBlock.x>>>(N, d_A, d_sum);
    cudaCheck("reduction_1_interleaved kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_1_interleaved kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Interleaved: warp divergence done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    // re- zero out d_sum
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");

    // Interleaved reduction kernel 2
    printf("%25s\n", "Interleaved: bank conflict");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_2_interleaved<<<dimGrid.x, dimBlock.x>>>(N, d_A, d_sum);
    cudaCheck("reduction_1_interleaved kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_2_interleaved kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Interleaved: bank conflict done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");


    //  kernel 3: Sequential 1
    printf("%25s\n", "Kernel: sequential 1");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_3_sequential<<<dimGrid.x, dimBlock.x>>>(N, d_A, d_sum);
    cudaCheck("reduction_3_sequential kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_3_sequential kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Reduction 3: Sequential 1 kernel done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");


    //  kernel 4: Sequential 2
    printf("%25s\n", "Kernel: sequential 2");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_4_sequential<<<dimGrid.x, dimBlock.x/2>>>(N, d_A, d_sum);
    cudaCheck("reduction_4_sequential kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_4_sequential kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Reduction 4: Sequential 2 kernel done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");


    //  kernel 5: Sequential 3
    printf("%25s\n", "Kernel: sequential 3");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_5_sequential<<<dimGrid.x/2, dimBlock.x>>>(N, d_A, d_sum);
    cudaCheck("reduction_5_sequential kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_5_sequential kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Reduction 5: Sequential 3 kernel done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");

    //  kernel 6: Sequential 4
    printf("%25s\n", "Kernel: sequential 3");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    reduction_6_sequential<<<dimGrid.x/2, dimBlock.x/2>>>(N, d_A, d_sum);
    cudaCheck("reduction_6_sequential kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("reduction_6_sequential kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "Reduction 6: Sequential 4 kernel done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");


    //  kernel 7: warp shuffle
    printf("%25s\n", "Kernel: Warp Shuffle");
    // record the event
    cudaEventRecord(startEvent, 0);
    cudaCheck("cudaEventRecord failure");
    // launch kernel
    ws_reduce<<<dimGrid.x/2, dimBlock.x/2>>>(N, d_A, d_sum);
    cudaCheck("ws_reduce kernel launch failure");
    cudaEventRecord(stopEvent, 0);
    cudaCheck("ws_reduce kernel execution failure of cudaEventRecord failure");
    printf("%25s\n", "ws_reduce kernel done");
    cudaEventSynchronize(stopEvent);
    cudaCheck("cudaEventSynchronize failure");
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaCheck("cudaEventElapsedTime failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("memcpy D2H failure");
    //check correctness, print timing and bandwidth
    postprocess(ref, h_sum, ms);
    cudaMemset(d_sum, 0.0f, sizeof(float));
    cudaCheck("cudaMemset failure");


    // cleanup
    // destroy events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    // free device pointers;
    cudaFree(d_A);
    cudaFree(d_sum);
    cudaCheck("cudaFree failure");
    delete[] h_A;
    delete h_sum;
    printf("%s\n", "Done, bye");
    return 0;
}

