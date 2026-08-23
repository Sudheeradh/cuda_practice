#include <cuda_runtime.h>

__global__ void final_reduce_post_process(float* d_partials, float* output, float eps, int N) {
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float stemp[];
    stemp[tid] = 0;

    for (int i = 0; i < N; i += block_dim) {
        if (tid + i < N) stemp[tid] += d_partials[tid + i];
    }
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            stemp[tid] += stemp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) *output = sqrtf(stemp[0] + eps);

}

__global__ void reduce(const float* input, float* d_partials, int N) {
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float stemp[];

    if (idx < N) {
        stemp[tid] = (input[idx] * input[idx]) / N;
    } else {
        stemp[tid] = 0;
    }

    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            stemp[tid] += stemp[tid + stride];
        }
        __syncthreads();
    }

    d_partials[block_idx] = stemp[0];
}

__global__ void rms(const float* input, float* d_rms, float gamma, float beta, float* output, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) output[idx] = (gamma * (input[idx] / *d_rms)) + beta;
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemBytes = threadsPerBlock * sizeof(float);

    float* d_partials = nullptr;
    float* d_rms = nullptr;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_rms, sizeof(float));

    reduce<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(input, d_partials, N);
    final_reduce_post_process<<<1, threadsPerBlock, sharedMemBytes>>>(d_partials, d_rms, eps, blocksPerGrid);

    rms<<<blocksPerGrid, threadsPerBlock>>>(input, d_rms, gamma, beta, output, N);

    cudaFree(d_partials);
    cudaFree(d_rms);
                      }
