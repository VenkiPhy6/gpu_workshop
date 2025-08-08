# Notes and code from "Accelerating Python with GPU" workshop

Venkatesh Subramanian

Aug 8, 2025

8:30-13:00


**Quick notes:**
1. For Numpy to run on multiple cores within a node, you need to set some env variable in the beginning of the SLURM script.
2. Minimize transfers between GPU and CPU. Transfers are costly
3. Pinned memory: Ask for a part of the RAM
4. Mapped memory: Pointer of the RAM memory for the GPU to use 
5. Cupy
   1. `cupy-cuda12x` drivers are needed in the dev. This is what is in Juno.
      1. Python 3.11 works. 3.12 seems to have issues. Don't know about 3.10.
   2. How much control can Python give?
      1. You can use mapped/pinned memory
      2. You can write kernel functions
6. Numba lets you treat Python like its compiled (JIT)
   1. `@jit` and `@cuda.jit` decorator
   2. `pip3 install numba` and then `from numba import jit`
      1. NVidia defined its own datatypes (`tf32`) to speed things up (instead of `float32`). This is activated by `fastmath` arg to the `@cuda.jit` decorator
7. TensorFlow can fallback to CPU if no GPU but Pytorch won't

