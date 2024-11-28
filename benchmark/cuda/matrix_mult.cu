#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {
    void matrix_multiply(const float* A, const float* B, float* C, int N) {
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, N * N * sizeof(float));
        cudaMalloc(&d_B, N * N * sizeof(float));
        cudaMalloc(&d_C, N * N * sizeof(float));
        
        cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
        
        matrix_multiply_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        
        cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}