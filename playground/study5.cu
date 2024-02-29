#include <cstdio>
#include <iostream>
#include <array>
#include <chrono>
#include <type_traits>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define RETURN_IF_ERROR(error_code) if (error_code != cudaSuccess) {LOG(ERROR) << "CUDA error, code:" << error_code << ", name: " << cudaGetErrorName(error_code);return error_code;}

#define RETURN_IF_NULL(p) if (p == nullptr) {LOG(ERROR) << "Host memory failed!"; exit( EXIT_FAILURE );}

class TimeCost {
private:
    std::chrono::time_point <std::chrono::system_clock> start;
public:
    TimeCost() : start(std::chrono::system_clock::now()) {}

    ~TimeCost() = default;

    auto Cost() {
        // us
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start);
        return duration.count();
    }

    auto reset() {
        start = std::chrono::system_clock::now();
    }
};

inline cudaError_t ErrorCheck(cudaError_t error_code, const std::string &filename, int lineNumber) {
    if (error_code != cudaSuccess) {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int *temp, int val) {
    int lane = g.thread_rank();
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val;
        g.sync();
        if (lane < i) val += temp[lane + i];
        g.sync();
    }
    return val;
}

__global__ void getThisBlock() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("index: %d\n", index);
    thread_block block = this_thread_block();

}

cudaError_t TestGetBlock() {
    getThisBlock<<<2, 2>>>();
    cudaDeviceSynchronize();
    return cudaSuccess;
}

__device__ int thread_sum(int *input, int n) {
    int sum = 0;
    for (int i = blockDim.x * blockIdx.x; i < n / 4; i += blockIdx.x * blockDim.x) {
        int4 in = ((int4 *) input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_kernel_block(int *sum, int *input, int n) {
    int my_sum = thread_sum(input, n);
    extern __shared__ int temp[];
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, my_sum);
    if (g.thread_rank() == 0) atomicAdd(sum, block_sum);
}

template<typename group_t>
__device__ int reduce_sum_2(group_t g, int *temp, int val) {
    int lane = g.thread_rank();

#pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val;
        g.sync();
        if (lane < i) val += temp[lane + i];
        g.sync();
    }
    return val;
}

inline int Align_up(const int block_size, const int num) {
    return (num + block_size - 1) / block_size;
}

// relu forward
template<typename T>
__global__ void ReLUForwardKernel(const int N, const T *X, T *Y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        Y[index] = X[index] <= 0 ? T(0) : X[index];
    }
}

// relu backward
template<typename T>
__global__ void ReLUBackwardKernel(const int N, const T *dY, const T *Y, T *dX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dX[i] = Y[i] > T(0) ? dY[i] : T(0);
    }
}

// Softplus forward
template<typename T>
__global__ void SoftPlusBackwardKernel(const int N, const T *X, T *Y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        Y[i] = log(exp(X[i]) + T(1));
    }
}

// Softplus backward
template<typename T>
__global__ void SoftPlusBackwardKernel(const int N, const T *dY, const T *Y, T *dX) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        const float nexpY = static_cast<float>(exp(-Y[i]));
        dY[i] = dY[i] * (T(1) - nexpY);
    }
}

// Softmax forward
template<typename T>
__global__ void SigmoidForwardKernel(const int N, const T *X, T *Y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        Y[i] = 1 / (1 + expf(X[i]));
    }
}

// Softmax backward
template<typename T>
__global__ void SigmoidBackwardKernel(const int N, const T *dY, const T *Y, T *dX) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        dX[i] = dY[i] * Y[i] * (T(1) - Y[i]);
    }
}

// tanh forward
template<typename T>
__global__ void TanhForwardKernel(const int N, const T *X, T *Y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        Y[i] = T(2) / (1 + expf(T(-2) * X[i])) - 1;
    }
}

// tanh backward
template<typename T>
__global__ void TanhBackwardKernel(const int N, const T *dY, const T *Y, T *dX) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = index; i < N; i += stride) {
        const float expX = expf(T(-2) * X[i]);
        dX[i] = 1 - pow(Y[i], 2);
    }
}

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)


int main() {
    TestGetBlock();
}