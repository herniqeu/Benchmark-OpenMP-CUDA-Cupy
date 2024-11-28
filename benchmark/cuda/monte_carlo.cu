#include <curand_kernel.h>

__global__ void monte_carlo_kernel(int n_points, int* inside, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        curand_init(1234, idx, 0, &states[idx]);
        float x = curand_uniform(&states[idx]);
        float y = curand_uniform(&states[idx]);
        if (x*x + y*y <= 1.0f) atomicAdd(inside, 1);
    }
}

extern "C" float monte_carlo_pi(int n_points) {
    int* d_inside;
    int h_inside = 0;
    curandState* d_states;
    
    cudaMalloc(&d_inside, sizeof(int));
    cudaMalloc(&d_states, n_points * sizeof(curandState));
    cudaMemcpy(d_inside, &h_inside, sizeof(int), cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (n_points + block_size - 1) / block_size;
    monte_carlo_kernel<<<num_blocks, block_size>>>(n_points, d_inside, d_states);
    
    cudaMemcpy(&h_inside, d_inside, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_inside);
    cudaFree(d_states);
    
    return 4.0f * h_inside / n_points;
}