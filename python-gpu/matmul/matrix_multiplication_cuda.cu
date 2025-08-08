// matrix_multiplication_cuda.cu
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    size_t size = static_cast<size_t>(N) * N;

    // Allocate host memory
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    // Initialize matrices with random values
    for (size_t i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size * sizeof(float)), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, size * sizeof(float)), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, size * sizeof(float)), "cudaMalloc C");

    // Copy host to device
    checkCuda(cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice), "memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice), "memcpy B");

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    // Matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "N=" << N << ", time=" << elapsed.count() << " seconds" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
