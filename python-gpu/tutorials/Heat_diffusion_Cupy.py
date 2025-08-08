import cupy as cp
#import matplotlib.pyplot as plt
import time

# Parameters
N = 256
tolerance = 1e-4
max_iters = 100000
dx = 1.0 / N

# Initialize grid on GPU
p = cp.zeros((N, N), dtype=cp.float32)
f = cp.zeros((N, N), dtype=cp.float32)
f[N//2, N//2] = 10000  # Heat source

start = time.time()
# Jacobi iteration
for iteration in range(max_iters):
####################################
#######Insert your code here########
####################################

    # Optional: bring scalar to CPU to check for convergence
    if diff.get() < tolerance:
        print(f"Converged after {iteration} iterations")
        break

    p = p_new

end = time.time()
# Transfer result back to CPU for plotting
p_cpu = cp.asnumpy(p)
print("Heat source at equilibrium: ", p_cpu[N//2, N//2])
print("Elapsed time: ", end - start, " s")

#plt.imshow(p_cpu, cmap='hot')
#plt.colorbar()
#plt.title("CuPy Heat Diffusion")
#plt.show()
