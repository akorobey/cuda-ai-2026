#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "naive_gemm_cuda.h"

__global__ void vecNaiveGemm4(const float* A, const float* B, float* C, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x * 4 < n && y < n) {
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

        for (int i = 0; i < n; ++i) {
            float a = A[y * n + i];
            float4 b = B4[(i * n / 4) + x]; 

            sum.x += a * b.x;
            sum.y += a * b.y;
            sum.z += a * b.z;
            sum.w += a * b.w;
        }

        C4[(y * n / 4) + x] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    int bytes = n * n * sizeof(float);
    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&C, bytes);
    
    cudaMemcpy(A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threads(32, 8);
    dim3 blocks(
        (n / 4 + threads.x - 1) / threads.x,
        (n + threads.y - 1) / threads/*  */.y
    );
    vecNaiveGemm4<<<blocks, threads>>>(A, B, C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return c;
}

#if 0
std::vector<float> NaiveGemmRef(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.f);

    const float* aptr = a.data();
    const float* bptr = b.data();
    float* cptr = c.data();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float aval = aptr[i * n + k];
            for (int j = 0; j < n; j++) {
                cptr[i * n + j] += aval * bptr[k * n + j];
            }
        }
    }

    return c;
}

int main() {
    size_t n = 4096;
    std::vector<float> a(n*n);
    std::vector<float> b(n*n);
    for (size_t i = 0; i < n*n; i++) {
        a[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
        b[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto c = NaiveGemmCUDA(a, b, n);

    auto cref = NaiveGemmRef(a, b, n);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(c[i] - cref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = NaiveGemmCUDA(a, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.4f\n", time);

    return 0;
}
#endif
