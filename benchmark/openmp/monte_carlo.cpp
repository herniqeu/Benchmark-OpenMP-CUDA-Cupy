#include <random>
#include <omp.h>

extern "C" double monte_carlo_pi(int n_points) {
    int inside = 0;
    
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        
        #pragma omp for reduction(+:inside)
        for(int i = 0; i < n_points; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if(x*x + y*y <= 1.0) inside++;
        }
    }
    
    return 4.0 * inside / (double)n_points;
}