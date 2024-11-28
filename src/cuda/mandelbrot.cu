__global__ void mandelbrot_kernel(unsigned char* image, int width, int height, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x < width && y < height) {
        const double x_min = -2.0;
        const double x_max = 1.0;
        const double y_min = -1.5;
        const double y_max = 1.5;
        
        double cr = x_min + (x_max - x_min) * x / width;
        double ci = y_min + (y_max - y_min) * y / height;
        double zr = 0.0, zi = 0.0;
        int iter;
        
        for(iter = 0; iter < max_iter; iter++) {
            double zr2 = zr * zr;
            double zi2 = zi * zi;
            if(zr2 + zi2 > 4.0) break;
            double zr_new = zr2 - zi2 + cr;
            zi = 2 * zr * zi + ci;
            zr = zr_new;
        }
        
        image[y * width + x] = iter * 255 / max_iter;
    }
}

extern "C" void mandelbrot(unsigned char* image, int width, int height, int max_iter) {
    unsigned char* d_image;
    cudaMalloc(&d_image, width * height);
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    mandelbrot_kernel<<<grid, block>>>(d_image, width, height, max_iter);
    
    cudaMemcpy(image, d_image, width * height, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}