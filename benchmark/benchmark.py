import json
import time
import ctypes
import numpy as np
from pathlib import Path

class Benchmark:
    def __init__(self):
        self.cuda_lib = ctypes.CDLL('./cuda/libmonte_carlo.so')
        self.openmp_lib = ctypes.CDLL('./openmp/libmonte_carlo.so')
        from cupy.monte_carlo import monte_carlo_pi as cupy_monte_carlo
        self.cupy_monte_carlo = cupy_monte_carlo
        
    def run_benchmark(self, points_list, iterations=5):
        results = {}
        
        for points in points_list:
            results[points] = {
                'cuda': [],
                'openmp': [],
                'cupy': []
            }
            
            for _ in range(iterations):
                # CUDA
                start = time.perf_counter()
                self.cuda_lib.monte_carlo_pi(points)
                results[points]['cuda'].append(time.perf_counter() - start)
                
                # OpenMP
                start = time.perf_counter()
                self.openmp_lib.monte_carlo_pi(points)
                results[points]['openmp'].append(time.perf_counter() - start)
                
                # CuPy
                start = time.perf_counter()
                self.cupy_monte_carlo(points)
                results[points]['cupy'].append(time.perf_counter() - start)
        
        return results
    
    def save_results(self, results, filename='benchmark_results.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    points_list = [1000, 10000, 100000, 1000000]
    benchmark = Benchmark()
    results = benchmark.run_benchmark(points_list)
    benchmark.save_results(results)