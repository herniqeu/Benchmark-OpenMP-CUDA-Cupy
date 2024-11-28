import cupy as cp

def monte_carlo_pi(n_points):
    x = cp.random.uniform(0, 1, n_points)
    y = cp.random.uniform(0, 1, n_points)
    inside = cp.sum((x*x + y*y) <= 1.0)
    pi = 4.0 * inside / n_points
    return float(pi)
