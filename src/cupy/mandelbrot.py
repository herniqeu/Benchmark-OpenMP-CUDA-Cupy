import cupy as cp

def mandelbrot(width, height, max_iter):
    x = cp.linspace(-2, 1, width)
    y = cp.linspace(-1.5, 1.5, height)
    c = x[:, None] + 1j * y[None, :]
    
    z = cp.zeros_like(c)
    divtime = max_iter + cp.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z*z + c
        diverge = z*cp.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    
    return cp.asnumpy(divtime * 255 / max_iter).astype(cp.uint8)