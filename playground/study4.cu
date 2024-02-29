#include <cstdio>
#include <iostream>
#include <array>
#include <chrono>
#include <type_traits>
#include <glog/logging.h>
#include <cuda_runtime.h>

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

template<typename T, typename std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
__global__ void relu_kernel1(T *input, T *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    return output[index] = input[index] < 0 ? 0 : input[index];
}

cudaError_t TestCudaStream() {
    cudaStream_t stream;
    RETURN_IF_ERROR(cudaStreamCreate(&stream));

    constexpr int N = 1 << 8;
    float *d_a, *a;

    a = static_cast<float *>(malloc(N));
    RETURN_IF_NULL(a);

    RETURN_IF_ERROR(cudaMalloc((void **) &d_a, N));
    TimeCost cost{};
    RETURN_IF_ERROR(cudaMemcpyAsync((void *) d_a, (void *) a, N, cudaMemcpyHostToDevice, stream));
    LOG(INFO) << "Async Copy Cost: " << cost.Cost() << " us";
    RETURN_IF_ERROR(cudaStreamDestroy(stream));
    return cudaSuccess;
}

__global__ void kernel_func1(float *a, int offset) {
    int index = blockDim.x * blockIdx.x + threadIdx.x + offset;
    auto x = static_cast<float>(a[index]);
    auto s = sinf(x), c = cosf(x);
    a[index] += sqrtf(s * s + c * s);
}

float maxError(float *a, int n) {
    float maxE = 0;
    for (int i = 0; i < n; i++) {
        float error = fabs(a[i] - 1.0f);
        if (error > maxE) maxE = error;
    }
    return maxE;
}

cudaError_t TestAsyncStream() {
    constexpr int blockSize = 256, nStreams = 4;
    constexpr int n = 4 * 1024 * blockSize * nStreams;
    constexpr int streamSize = n / nStreams;
    constexpr int streamBytes = streamSize * sizeof(float);
    constexpr int bytes = n * sizeof(float);
    int devId = 0;

    cudaDeviceProp prop;
    RETURN_IF_ERROR(cudaGetDeviceProperties(&prop, devId));
    printf("Device : %s\n", prop.name);
    RETURN_IF_ERROR(cudaSetDevice(devId));

    float *a, *d_a;
    RETURN_IF_ERROR(cudaMallocHost((void **) &a, bytes));
    RETURN_IF_ERROR(cudaMalloc((void **) &d_a, bytes));

    // 1. sequence
    TimeCost cost{};
    memset((void *) a, 0, bytes);
    RETURN_IF_ERROR(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    kernel_func1<<<n / blockSize, blockSize>>>(d_a, 0);
    RETURN_IF_ERROR(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    LOG(INFO) << "Seq Cost: " << cost.Cost() << " us";

    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        RETURN_IF_ERROR(cudaStreamCreate(&stream[i]));
    }

    // 2. Async 1
    cost.reset();
    memset((void *) a, 0, bytes);
    for (int i = 0; i < nStreams; ++i) {
        int offset = streamSize * i;
        RETURN_IF_ERROR(cudaMemcpyAsync(&d_a[offset], &a[offset], streamSize, cudaMemcpyHostToDevice, stream[i]));
        kernel_func1<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        RETURN_IF_ERROR(cudaMemcpyAsync(&a[offset], &d_a[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]));
    }
    cudaDeviceSynchronize();
    LOG(INFO) << "Async1 Cost: " << cost.Cost() << " us";

    // 3. Async 2
    cost.reset();
    memset((void *) a, 0, bytes);
    for (int i = 0; i < nStreams; ++i) {
        int offset = streamSize * i;
        RETURN_IF_ERROR(cudaMemcpyAsync(&d_a[offset], &a[offset], streamSize, cudaMemcpyHostToDevice, stream[i]));
    }
    for (int i = 0; i < nStreams; ++i) {
        int offset = streamSize * i;
        kernel_func1<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    }
    for (int i = 0; i < nStreams; ++i) {
        int offset = streamSize * i;
        RETURN_IF_ERROR(cudaMemcpyAsync(&a[offset], &d_a[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]));
    }
    cudaDeviceSynchronize();
    LOG(INFO) << "Async2 Cost: " << cost.Cost() << " us";

    RETURN_IF_ERROR(cudaFreeHost(a));
    RETURN_IF_ERROR(cudaFree(d_a));
    for (int i = 0; i < nStreams; ++i) {
        RETURN_IF_ERROR(cudaStreamDestroy(stream[i]));
    }
    return cudaSuccess;
}

// shared memory
__global__ void staticReverse(int *d, int n) {
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}


__global__ void dynamicReverse(int *d, int n) {
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

cudaError_t TestReverse() {
    const int n = 64;
    int a[n], r[n], d[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n - i - 1;
        d[i] = 0;
    }
    int *d_d;
    RETURN_IF_ERROR(cudaMalloc((void **) &d_d, n * sizeof(int)));
    RETURN_IF_ERROR(cudaMemcpy(d_d, d, n * sizeof(int), cudaMemcpyHostToDevice));
    staticReverse<<<1, n>>>(d_d, n);
    RETURN_IF_ERROR(cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost));

    RETURN_IF_ERROR(cudaMemcpy(d_d, d, n * sizeof(int), cudaMemcpyHostToDevice));
    dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
    RETURN_IF_ERROR(cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(cudaFree(d_d));

    return cudaSuccess;
}

// pinned memory
cudaError_t TestPinnedMemory(){
    int * pinned_data, * portable_data;
    constexpr int N = 1024;
    // use pinned memory
    RETURN_IF_ERROR(cudaMallocHost((void**)&pinned_data,N*sizeof(int)));
    // don't use pinned memory
    portable_data = static_cast<int*>(malloc(N * sizeof(int)));
    return cudaSuccess;
}

// Spaxy
__global__ void spaxy(int n, int a, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) y[index] = a * x[index] + x[index];
}

cudaError_t TestSpaxy(){
    constexpr int N = 1 << 20;
    constexpr int block_size = 256;
    constexpr int grid_size = static_cast<int>((N + block_size -  1) / block_size);
    constexpr int a = 6;
    float *x, *y, *d_x, *d_y;
    cudaMallocHost(&x, N * sizeof (float ));
    cudaMallocHost(&y, N * sizeof (float ));
    cudaMalloc(&d_x, N * sizeof(float ));
    cudaMalloc(&d_y, N * sizeof(float ));
    memset(x, 0, N);
    memset(y, 0, N);
    cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);
    spaxy<<<grid_size, block_size>>>(N, 6, d_x, d_y);
    cudaMemcpy(x, d_x, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFreeHost(x);
    cudaFreeHost(y);
    return cudaSuccess;
}



int main() {
    TestCudaStream();

    TestAsyncStream();

    TestReverse();

    TestPinnedMemory();

    TestSpaxy();
}
