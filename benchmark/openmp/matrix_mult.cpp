#include <omp.h>

extern "C" {
    void matrix_multiply(const float* A, const float* B, float* C, int N) {
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                float sum = 0.0f;
                for(int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}