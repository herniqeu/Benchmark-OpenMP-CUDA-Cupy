import cupy as cp

def nbody_simulate(pos, vel, mass, dt):
    G = 1.0
    n_bodies = len(mass)
    
    pos = cp.asarray(pos)
    vel = cp.asarray(vel)
    mass = cp.asarray(mass)
    
    for i in range(n_bodies):
        r = pos.reshape(-1, 3) - pos[i]
        dist = cp.linalg.norm(r, axis=1)
        dist[i] = 1.0  #