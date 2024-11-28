#include <omp.h>
#include <cmath>

extern "C" void nbody_simulate(float* pos, float* vel, float* mass, int n_bodies, float dt) {
    const float G = 1.0f;
    
    #pragma omp parallel for
    for(int i = 0; i < n_bodies; i++) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        for(int j = 0; j < n_bodies; j++) {
            if(i != j) {
                float dx = pos[j*3] - pos[i*3];
                float dy = pos[j*3+1] - pos[i*3+1];
                float dz = pos[j*3+2] - pos[i*3+2];
                float dist = sqrt(dx*dx + dy*dy + dz*dz);
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