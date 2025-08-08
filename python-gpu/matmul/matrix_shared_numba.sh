#!/bin/bash
#SBATCH --partition=h100
#SBATCH --mem=4GB                         # Memory 
#SBATCH --job-name=matmul_numbagpu              # Job Name
#SBATCH -o LOG_sharednumbagpu                            # Log file
#SBATCH --time=00:20:00                      # WallTime
#SBATCH --nodes=1                           # Number of Nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks (MPI presseces)
#SBATCH --cpus-per-task=1                    # Number of processors per task OpenMP threads()
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL                      # Add all jobs to the the mailing list
#SBATCH --mail-user=dal281726@utdallas.edu       # Send notification to the address when job begins and ends

module purge
module load gnu12
module load python/3.11.11
module load cuda/12.6
source /home/dal281726/accelerating-python-with-gpus/cupy_venv/bin/activate
#pip3 install numba
for N in 512 1024 2048 4096 8192; do
    echo "Running GPU Numba with shared memory matrix multiplication with N=$N"
    python3 matrix_multiplication_shared_numba.py $N
done
