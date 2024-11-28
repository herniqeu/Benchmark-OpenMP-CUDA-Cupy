__global__ void nbody_kernel(float* pos, float* vel, float* mass, int n_bodies, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float G = 1.0f;
    
    if (i < n_bodies) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        for(int j = 0; j < n_bodies; j++) {
            if(i != j) {
                float dx = pos[j*3] - pos[i*3];
                float dy = pos[j*3+1] - pos[i*3+1];
                float dz = pos[j*3+2] - pos[i*3+2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                float f = G * mass[i] * mass[j] / (dist * dist * dist);
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        
        vel[i*3] += fx * dt / mass[i];
        vel[i*3+1] += fy * dt / mass[i];
        vel[i*3+2] += fz * dt / mass[i];
        
        pos[i*3] += vel[i*3] * dt;
        pos[i*3+1] += vel[i*3+1] * dt;
        pos[i*3+2] += vel[i*3+2] * dt;
    }
}

extern "C" void nbody_simulate(float* pos, float* vel, float* mass, int n_bodies, float dt) {
    float *d_pos, *d_vel, *d_mass;
    
    cudaMalloc(&d_pos, n_bodies * 3 * sizeof(float));
    cudaMalloc(&d_vel, n_bodies * 3 * sizeof(float));
    cudaMalloc(&d_mass, n_bodies * sizeof(float));
    
    cudaMemcpy(d_pos, pos, n_bodies * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, vel, n_bodies * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, n_bodies * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n_bodies + blockSize - 1) / blockSize;
    
    nbody_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_mass, n_bodies, dt);
    
    cudaMemcpy(pos, d_pos, n_bodies * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vel, d_vel, n_bodies * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
}