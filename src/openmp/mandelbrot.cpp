#include <omp.h>

extern "C" void mandelbrot(unsigned char* image, int width, int height, int max_iter) {
    const double x_min = -2.0;
    const double x_max = 1.0;
    const double y_min = -1.5;
    const double y_max = 1.5;
    
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
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
}