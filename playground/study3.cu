#include <cstdio>
#include <iostream>
#include <array>
#include <chrono>
#include <glog/logging.h>
#include <cuda_runtime.h>

#define RETURN_IF_ERROR(error_code) if (error_code != cudaSuccess) {LOG(ERROR) << "CUDA error, code:" << error_code << ", name: " << cudaGetErrorName(error_code);return error_code;}

#define RETURN_IF_NULL(p) if (p == nullptr) {LOG(ERROR) << "Host memory failed!"; exit( EXIT_FAILURE );}

cudaError_t ErrorCheck(cudaError_t error_code, const std::string &filename, int lineNumber) {
    if (error_code != cudaSuccess) {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

__global__ void print_grid_block_info_kernel() {
    printf("Grid id:%d, Block id: %d. Number of blocks in one grid: %d. "
           "Thread id: %d. Number of threads in one block: %d\n",
           gridDim.x, blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

cudaError_t TestPrintInfo() {
    constexpr int GRID_SIZE = 4;
    constexpr int BLOCK_SIZE = 3;
    print_grid_block_info_kernel<<<GRID_SIZE, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
    RETURN_IF_ERROR(cudaGetLastError());
}

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

__global__ void pointwise_add_kernel1(int out[], int a[], int b[], int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

__global__ void pointwise_add_kernel2(int out[], int a[], int b[], int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}


// stream
void TestCUDAStream() {
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&stream[i]);
    }
}

// Sum of 2D Matrix
void sumMatrix2D_CPU(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    auto *a = MatA;
    auto *b = MatB;
    auto *c = MatC;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            c[i] = a[i] + b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

__global__ void sumMatrix(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix + iy * ny;
    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

cudaError_t Test2DMatrix() {
    int nx = 1 << 12;
    int ny = 1 << 12;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *A_host = (float *) malloc(nBytes);
    float *B_host = (float *) malloc(nBytes);
    float *C_host = (float *) malloc(nBytes);
    float *C_from_gpu = (float *) malloc(nBytes);

    //cudaMalloc
    float *A_dev = NULL;
    float *B_dev = NULL;
    float *C_dev = NULL;
    RETURN_IF_ERROR(cudaMalloc((void **) &A_dev, nBytes));
    RETURN_IF_ERROR(cudaMalloc((void **) &B_dev, nBytes));
    RETURN_IF_ERROR(cudaMalloc((void **) &C_dev, nBytes));

    RETURN_IF_ERROR(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    RETURN_IF_ERROR(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    TimeCost cost{};
    sumMatrix<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);
    LOG(INFO) << "Cost: " << cost.Cost() << " us";
    return cudaSuccess;
}

__device__ float devData;
__constant__ float constVar;

__global__ void checkGlobalVariable() {
    printf("Device: The value of the global variable is %f\n", devData);
    devData += 2.0;
}

// Symbol memory used for global vars, defined with the __device__, __constant__, or __managed__ qualifiers in CUDA C/C++ code
void TestSymbol() {
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    cudaMemcpyToSymbol(constVar, &value, sizeof(float));
    printf("Host: copy %f to the global variable\n", value);
    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f \n", value);

}

// Zero Copy
__global__ void sumArrayCUDA(float *res, float *a, float *b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    res[index] = a[index] + b[index];
}

cudaError_t TestZeroCopy() {
    int nElem = 1 << 8;
    dim3 block(1024);
    dim3 grid(nElem / block.x);
    float host_nums[nElem];
    float host_gpu_nums[nElem];
    memset(host_nums, 0, nElem);
    memset(host_gpu_nums, 0, nElem);
    float *a_host,*b_host,*res_d;
    RETURN_IF_ERROR(cudaMalloc((float**)&res_d,nElem));
    float *a_dev, *b_dev;
    RETURN_IF_ERROR(cudaHostAlloc((float **)&a_host, nElem, cudaHostAllocMapped));
    RETURN_IF_ERROR(cudaHostAlloc((float **)&b_host, nElem, cudaHostAllocMapped));
    RETURN_IF_ERROR(cudaHostGetDevicePointer((void**)&a_dev, (void*)a_host, 0));
    RETURN_IF_ERROR(cudaHostGetDevicePointer((void**)&b_dev, (void*)b_host, 0));
    sumArrayCUDA<<<grid, block>>>(res_d, a_dev, b_dev);
    return cudaSuccess;
}

int main() {
    TestPrintInfo();

    Test2DMatrix();

    TestSymbol();

    TestZeroCopy();
}