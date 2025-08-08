import numpy as np
import time
#import matplotlib.pyplot as plt

# Parameters
N = 256
tolerance = 1e-4
max_iters = 100000
dx = 1.0 / N

# Initialize grid
p = np.zeros((N, N))
f = np.zeros((N, N))
f[N//2, N//2] = 10000  # Heat source

# Boundary conditions: p = 0 on the edges (already zero)

# Jacobi iteration
start = time.time()
for iteration in range(max_iters):
    p_new = p.copy()
    p_new[1:-1, 1:-1] = 0.25 * (
        p[:-2, 1:-1] + p[2:, 1:-1] +
        p[1:-1, :-2] + p[1:-1, 2:] +
        dx**2 * f[1:-1, 1:-1]
    )
    diff = np.linalg.norm(p_new - p)
    p = p_new
    if diff < tolerance:
        print(f"Converged after {iteration} iterations")
        break
end = time.time()    
print("Heat source at equilibrium: ", p[N//2, N//2])
print("Elapsed time: ", end - start, " s")

#plt.imshow(p, cmap='hot')
#plt.colorbar()
#plt.title("NumPy Heat Diffusion")
#plt.show()
