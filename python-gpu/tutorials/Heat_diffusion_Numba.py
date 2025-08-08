import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time

# Parameters
N = 256
tolerance = 1e-4
max_iters = 100000
dx = 1.0 / N

# Initialize grid
p = np.zeros((N, N), dtype=np.float32)
f = np.zeros((N, N), dtype=np.float32)
f[N//2, N//2] = 10000  # Heat source

########### You start here ############

@njit(parallel=True)
def jacobi_solver(p, f, dx, tolerance, max_iters):
    N = p.shape[0]
    for iteration in range(max_iters):
        p_new = np.copy(p)
       
       #########################################
       ######### Write your code here ##########
       #########################################

        diff = np.linalg.norm(p_new - p)
        if diff < tolerance:
            print(f"Converged after {iteration} iterations")
            return p_new

        p = p_new

    print("Did not converge within the maximum number of iterations.")
    return p

############ No futher adjustment needed ############## 

# Run solver
start = time.time()
p = jacobi_solver(p, f, dx, tolerance, max_iters)
end = time.time()
print("Heat source at equilibrium: ", p[N//2, N//2])
print("Elapsed time: ", end - start, " s")

# Plotting
#plt.imshow(p, cmap='hot')
#plt.colorbar()
#plt.title("Numba Jacobi Heat Diffusion")
#plt.show()
