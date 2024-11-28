import json
import time
import ctypes
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Adiciona o diretório 'cupy' ao path do Python
sys.path.append('./cupy')

class Benchmark:
    def __init__(self):
        self.cuda_lib = ctypes.CDLL('./cuda/libmonte_carlo.so')
        self.openmp_lib = ctypes.CDLL('./openmp/libmonte_carlo.so')
        from monte_carlo import monte_carlo_pi
        self.cupy_monte_carlo = monte_carlo_pi
        
    def run_benchmark(self, points_list, iterations=5):
        results = {}
        
        # Barra de progresso principal para diferentes tamanhos de pontos
        for points in tqdm(points_list, desc="Processing different point sizes"):
            results[points] = {
                'cuda': [],
                'openmp': [],
                'cupy': []
            }
            
            # Barra de progresso para as iterações
            for _ in tqdm(range(iterations), desc=f"Running iterations for {points} points", leave=False):
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
        print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    points_list = [
    10,           # 10 pontos
    100,          # 100 pontos
    1_000,        # mil
    10_000,       # 10 mil
    100_000,      # 100 mil
    1_000_000,    # 1 milhão
    10_000_000,   # 10 milhões
    ]
    print("Starting benchmark...")
    benchmark = Benchmark()
    results = benchmark.run_benchmark(points_list)
    benchmark.save_results(results)
