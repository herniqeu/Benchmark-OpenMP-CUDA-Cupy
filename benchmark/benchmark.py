import json
import time
import ctypes
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import platform
import psutil
import cpuinfo
import os
import datetime
import GPUtil

sys.path.append('./cupy')

class Benchmark:
    def __init__(self):
        self.cuda_lib = ctypes.CDLL('./cuda/libmonte_carlo.so')
        self.openmp_lib = ctypes.CDLL('./openmp/libmonte_carlo.so')
        self.cuda_lib.monte_carlo_pi.argtypes = [ctypes.c_int]
        self.cuda_lib.monte_carlo_pi.restype = ctypes.c_float
        self.openmp_lib.monte_carlo_pi.argtypes = [ctypes.c_int]
        self.openmp_lib.monte_carlo_pi.restype = ctypes.c_double
        from monte_carlo import monte_carlo_pi
        self.cupy_monte_carlo = monte_carlo_pi
        
    def get_system_info(self):
        gpu = GPUtil.getGPUs()[0]
        return {
            "cpu": {
                "model": cpuinfo.get_cpu_info()['brand_raw'],
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().max,
                "memory_total": psutil.virtual_memory().total / (1024**3),
                "memory_available": psutil.virtual_memory().available / (1024**3)
            },
            "gpu": {
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_free": gpu.memoryFree,
                "temperature": gpu.temperature,
                "uuid": gpu.uuid
            },
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version()
            }
        }

    def run_benchmark(self, points_list, iterations=5):
        results = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_info": self.get_system_info(),
                "benchmark_params": {
                    "points_list": [int(p) for p in points_list],
                    "iterations": iterations
                }
            },
            "results": {}
        }
        
        for points in tqdm(points_list, desc="Processing different point sizes"):
            points = int(points)
            results["results"][str(points)] = {
                "cuda": {
                    "times": [],
                    "memory_usage": [],
                    "gpu_temp": [],
                    "gpu_util": []
                },
                "openmp": {
                    "times": [],
                    "cpu_util": [],
                    "memory_usage": []
                },
                "cupy": {
                    "times": [],
                    "memory_usage": [],
                    "gpu_temp": [],
                    "gpu_util": []
                }
            }
            
            for _ in tqdm(range(iterations), desc=f"Running iterations for {points} points", leave=False):
                gpu = GPUtil.getGPUs()[0]
                
                # CUDA
                start = time.perf_counter()
                self.cuda_lib.monte_carlo_pi(points)
                cuda_time = time.perf_counter() - start
                results["results"][str(points)]["cuda"]["times"].append(cuda_time)
                results["results"][str(points)]["cuda"]["memory_usage"].append(gpu.memoryUsed)
                results["results"][str(points)]["cuda"]["gpu_temp"].append(gpu.temperature)
                results["results"][str(points)]["cuda"]["gpu_util"].append(gpu.load * 100)
                
                # OpenMP
                start = time.perf_counter()
                self.openmp_lib.monte_carlo_pi(points)
                openmp_time = time.perf_counter() - start
                results["results"][str(points)]["openmp"]["times"].append(openmp_time)
                results["results"][str(points)]["openmp"]["cpu_util"].append(psutil.cpu_percent())
                results["results"][str(points)]["openmp"]["memory_usage"].append(psutil.Process().memory_info().rss / 1024**2)
                
                # CuPy
                start = time.perf_counter()
                self.cupy_monte_carlo(points)
                cupy_time = time.perf_counter() - start
                results["results"][str(points)]["cupy"]["times"].append(cupy_time)
                results["results"][str(points)]["cupy"]["memory_usage"].append(gpu.memoryUsed)
                results["results"][str(points)]["cupy"]["gpu_temp"].append(gpu.temperature)
                results["results"][str(points)]["cupy"]["gpu_util"].append(gpu.load * 100)
                
                time.sleep(0.1)
        
        self.add_statistics(results)
        return results
    
    def add_statistics(self, results):
        for points in results["results"]:
            for framework in ["cuda", "openmp", "cupy"]:
                times = results["results"][points][framework]["times"]
                results["results"][points][framework]["statistics"] = {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "median": float(np.median(times))
                }
    
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
        100_000_000   # 100 milhões
    ]
    print("Starting benchmark...")
    benchmark = Benchmark()
    results = benchmark.run_benchmark(points_list)
    benchmark.save_results(results)