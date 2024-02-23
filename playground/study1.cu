#include <cstdio>
#include <iostream>
#include <array>
#include <chrono>
#include <glog/logging.h>

#define RETURN_IF_ERROR(error_code) if (error_code != cudaSuccess) {LOG(ERROR) << "CUDA error:\r\ncode=" << error_code << ", name=" << cudaGetErrorName(error_code);return error_code;}

#define RETURN_IF_NULL(p) if (p == nullptr) {LOG(ERROR) << "Host memory failed!"; exit( EXIT_FAILURE );}


__global__ void hello_from_gpu() {
    printf("Hello World from the the GPU\n");
    printf("How are you ?\n");
    const auto bid = blockIdx.x;
    const auto tid = threadIdx.x;
    const auto idx = tid + bid * blockDim.x;
    printf("bid:%d, tid:%d, idx:%d\n", bid, tid, idx);
}

cudaError_t ErrorCheck(cudaError_t error_code, const std::string &filename, int lineNumber) {
    if (error_code != cudaSuccess) {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void HelloFromGPU() {
    hello_from_gpu<<<1, 4>>>();
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    LOG(INFO) << "Hello, World!";
}


cudaError_t TestCudaDeviceCount() {
    int count;
    RETURN_IF_ERROR(cudaGetDeviceCount(&count));
    LOG(INFO) << "Cuda device count: " << count;
    return cudaSuccess;
}

cudaError_t TestCudaDeviceProp() {
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    int count;
    RETURN_IF_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        RETURN_IF_ERROR(cudaGetDeviceProperties(&prop, 0));
        LOG(INFO) << "Cuda device name: " << prop.name;
    }
    return cudaSuccess;
}

constexpr int N = 33 * 1024;

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

// used for device
template<typename T>
__global__ void vector_add(T *out, T *a, T *b, int n) {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type!");
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

template<typename T>
__global__ void vector_add2(T *out, T *a, T *b, int n) {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type!");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        out[tid] = a[tid] + b[tid];
        tid += blockDim.x + gridDim.x;
    }
}

cudaError_t TestCudaVectorAdd() {
    // 1. allocate host and device memory
    int a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = i * i;
        b[i] = -i;
    }

    int *d_a, *d_b, *d_c;
    RETURN_IF_ERROR(cudaMalloc((void **) &d_a, sizeof(int) * N));
    RETURN_IF_ERROR(cudaMalloc((void **) &d_b, sizeof(int) * N));
    RETURN_IF_ERROR(cudaMalloc((void **) &d_c, sizeof(int) * N));

    // 2. copy data from host to device
    RETURN_IF_ERROR(cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
    RETURN_IF_ERROR(cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));
    RETURN_IF_ERROR(cudaMemcpy(d_c, c, sizeof(int) * N, cudaMemcpyHostToDevice));

    // 3. launch kernel
    const int thread_blocks = 2, thread_per_block = 4;
    TimeCost cost{};
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    std::cout << "Cost baseline: " << cost.Cost() << " us" << std::endl;
    cost.reset();

    vector_add<<<thread_blocks, thread_per_block>>>(d_c, d_a, d_b, N);
    std::cout << "Cost func1: " << cost.Cost() << " us" << std::endl;
    cost.reset();

    vector_add2<<<thread_blocks, thread_per_block>>>(d_c, d_a, d_b, N);
    std::cout << "Cost func2: " << cost.Cost() << " us" << std::endl;

    cost.reset();
    vector_add2<<<thread_blocks * 2, thread_per_block * 2>>>(d_c, d_a, d_b, N);
    std::cout << "Cost func3: " << cost.Cost() << " us" << std::endl;

    // 4. copy result from device to host
    RETURN_IF_ERROR(cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost));
    std::cout << "\n";

    // 5. free device and host memory
    RETURN_IF_ERROR(cudaFree(d_a));
    RETURN_IF_ERROR(cudaFree(d_b));
    RETURN_IF_ERROR(cudaFree(d_c));
    return cudaSuccess;
}


int main() {

    // TestCudaDeviceCount();

    // TestCudaDeviceProp();

    TestCudaVectorAdd();

    return 0;
}
