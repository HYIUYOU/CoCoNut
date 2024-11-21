#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>



// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

void run_gemm_experiment(int N) {
    // 初始化 cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 初始化 alpha 和 beta
    float alpha = 1.0f;
    float beta = 0.1f;

    // 在 GPU 上分配 A, B 和 C
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    // 初始化 A, B 和 C，使用随机值
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 将 A, B 复制到 GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    // 执行多次实验并计算平均时间
    int iterations = 150000;
    std::vector<double> times(iterations);

    for (int i = 0; i < iterations; ++i) {
        cudaDeviceSynchronize(); // 同步 GPU

        auto start = std::chrono::high_resolution_clock::now();

        // 进行 GeMM 计算：C = alpha * A * B + beta * C
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

        cudaDeviceSynchronize(); // 确保计算完成
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times[i] = elapsed.count();
    }

    // 计算平均时间
    double average_time = 0.0;
    for (double t : times) {
        average_time += t;
    }
    average_time /= iterations;

    std::cout << "Average Time for GeMM with matrix size " << N << "x" << N << ": " << average_time << " seconds" << std::endl;

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

int main() {
    // 指定程序在 GPU 0 上运行
    CUDA_CHECK(cudaSetDevice(0));

    // 运行实验
    run_gemm_experiment(1024);

    return 0;
}
