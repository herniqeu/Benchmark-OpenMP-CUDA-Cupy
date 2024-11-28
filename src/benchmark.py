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
        # Carrega as bibliotecas
        self.cuda_lib = ctypes.CDLL('./cuda/libmonte_carlo.so')
        self.openmp_lib = ctypes.CDLL('./openmp/libmonte_carlo.so')
        self.cuda_matrix = ctypes.CDLL('./cuda/libmatrix_mult.so')
        self.openmp_matrix = ctypes.CDLL('./openmp/libmatrix_mult.so')
        self.cuda_nbody = ctypes.CDLL('./cuda/libnbody.so')
        self.openmp_nbody = ctypes.CDLL('./openmp/libnbody.so')
        self.cuda_mandelbrot = ctypes.CDLL('./cuda/libmandelbrot.so')
        self.openmp_mandelbrot = ctypes.CDLL('./openmp/libmandelbrot.so')

        # Importa implementações CuPy
        from monte_carlo import monte_carlo_pi
        from matrix_mult import matrix_multiply
        from nbody import nbody_simulate
        from mandelbrot import mandelbrot
        
        self.cupy_monte_carlo = monte_carlo_pi
        self.cupy_matrix = matrix_multiply
        self.cupy_nbody = nbody_simulate
        self.cupy_mandelbrot = mandelbrot
        
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

    def run_monte_carlo(self, points, results):
        gpu = GPUtil.getGPUs()[0]
        
        # CUDA
        start = time.perf_counter()
        self.cuda_lib.monte_carlo_pi(points)
        results['cuda']['times'].append(time.perf_counter() - start)
        
        # OpenMP
        start = time.perf_counter()
        self.openmp_lib.monte_carlo_pi(points)
        results['openmp']['times'].append(time.perf_counter() - start)
        
        # CuPy
        start = time.perf_counter()
        self.cupy_monte_carlo(points)
        results['cupy']['times'].append(time.perf_counter() - start)

    def run_matrix_mult(self, size, results):
        # Prepara dados
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)

        # CUDA
        start = time.perf_counter()
        self.cuda_matrix.matrix_multiply(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       ctypes.c_int(size))
        results['cuda']['times'].append(time.perf_counter() - start)

        # OpenMP
        start = time.perf_counter()
        self.openmp_matrix.matrix_multiply(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         ctypes.c_int(size))
        results['openmp']['times'].append(time.perf_counter() - start)

        # CuPy
        start = time.perf_counter()
        self.cupy_matrix(A, B)
        results['cupy']['times'].append(time.perf_counter() - start)

    def run_nbody(self, n_bodies, results):
        # Prepara dados
        pos = np.random.rand(n_bodies, 3).astype(np.float32)
        vel = np.zeros((n_bodies, 3), dtype=np.float32)
        mass = np.random.rand(n_bodies).astype(np.float32)
        dt = 0.01

        # CUDA
        start = time.perf_counter()
        self.cuda_nbody.nbody_simulate(pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     ctypes.c_int(n_bodies),
                                     ctypes.c_float(dt))
        results['cuda']['times'].append(time.perf_counter() - start)

        # OpenMP
        start = time.perf_counter()
        self.openmp_nbody.nbody_simulate(pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       mass.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                       ctypes.c_int(n_bodies),
                                       ctypes.c_float(dt))
        results['openmp']['times'].append(time.perf_counter() - start)

        # CuPy
        start = time.perf_counter()
        self.cupy_nbody(pos, vel, mass, dt)
        results['cupy']['times'].append(time.perf_counter() - start)

    def run_mandelbrot(self, size, results):
        width = height = size
        max_iter = 1000
        image = np.zeros(width * height, dtype=np.uint8)

        # CUDA
        start = time.perf_counter()
        self.cuda_mandelbrot.mandelbrot(image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                      ctypes.c_int(width),
                                      ctypes.c_int(height),
                                      ctypes.c_int(max_iter))
        results['cuda']['times'].append(time.perf_counter() - start)

        # OpenMP
        start = time.perf_counter()
        self.openmp_mandelbrot.mandelbrot(image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                        ctypes.c_int(width),
                                        ctypes.c_int(height),
                                        ctypes.c_int(max_iter))
        results['openmp']['times'].append(time.perf_counter() - start)

        # CuPy
        start = time.perf_counter()
        self.cupy_mandelbrot(width, height, max_iter)
        results['cupy']['times'].append(time.perf_counter() - start)

    def run_benchmark(self, points_list, matrix_sizes, nbody_sizes, mandelbrot_sizes, iterations=5):
        results = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "system_info": self.get_system_info(),
                "benchmark_params": {
                    "monte_carlo_points": points_list,
                    "matrix_sizes": matrix_sizes,
                    "nbody_sizes": nbody_sizes,
                    "mandelbrot_sizes": mandelbrot_sizes,
                    "iterations": iterations
                }
            },
            "results": {
                "monte_carlo": {},
                "matrix_mult": {},
                "nbody": {},
                "mandelbrot": {}
            }
        }

        # Monte Carlo
        for points in tqdm(points_list, desc="Running Monte Carlo"):
            results["results"]["monte_carlo"][points] = {
                "cuda": {"times": []},
                "openmp": {"times": []},
                "cupy": {"times": []}
            }
            for _ in range(iterations):
                self.run_monte_carlo(points, results["results"]["monte_carlo"][points])

        # Matrix Multiplication
        for size in tqdm(matrix_sizes, desc="Running Matrix Multiplication"):
            results["results"]["matrix_mult"][size] = {
                "cuda": {"times": []},
                "openmp": {"times": []},
                "cupy": {"times": []}
            }
            for _ in range(iterations):
                self.run_matrix_mult(size, results["results"]["matrix_mult"][size])

        # N-Body
        for size in tqdm(nbody_sizes, desc="Running N-Body"):
            results["results"]["nbody"][size] = {
                "cuda": {"times": []},
                "openmp": {"times": []},
                "cupy": {"times": []}
            }
            for _ in range(iterations):
                self.run_nbody(size, results["results"]["nbody"][size])

        # Mandelbrot
        for size in tqdm(mandelbrot_sizes, desc="Running Mandelbrot"):
            results["results"]["mandelbrot"][size] = {
                "cuda": {"times": []},
                "openmp": {"times": []},
                "cupy": {"times": []}
            }
            for _ in range(iterations):
                self.run_mandelbrot(size, results["results"]["mandelbrot"][size])

        self.add_statistics(results)
        return results

    def add_statistics(self, results):
        for algorithm in results["results"]:
            for size in results["results"][algorithm]:
                for framework in ["cuda", "openmp", "cupy"]:
                    times = results["results"][algorithm][size][framework]["times"]
                    results["results"][algorithm][size][framework]["statistics"] = {
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
    # Parâmetros para cada algoritmo
    points_list = [1000, 10000, 100000, 1000000, 10000000]
    matrix_sizes = [128, 256, 512, 1024, 2048]
    nbody_sizes = [1024, 2048, 4096, 8192]
    mandelbrot_sizes = [512, 1024, 2048, 4096]

    print("Starting benchmark...")
    benchmark = Benchmark()
    results = benchmark.run_benchmark(points_list, matrix_sizes, nbody_sizes, mandelbrot_sizes)
    benchmark.save_results(results)