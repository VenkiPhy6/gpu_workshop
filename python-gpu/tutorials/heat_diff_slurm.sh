#!/bin/bash
#SBATCH --partition=a30
#SBATCH --mem=16GB                         # Memory 
#SBATCH --job-name=dp_ddp              # Job Name
#SBATCH -o LOG_pytorch                            # Log file
#SBATCH --time=01:00:00                      # WallTime
#SBATCH --nodes=1                           # Number of Nodes
#SBATCH --ntasks=2                 # Number of tasks (MPI presseces)
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL                      # Add all jobs to the the mailing list
# SBATCH --mail-user=vxs210125@utdallas.edu       # Send notification to the address when job begins and ends

module purge
module load gnu12
module load python/3.11.11
module load cuda/12.6
source /home/vxs210125/work/gpu_workshop/py_gpuenv/bin/activate
#python3 benchmark_dp_ddp.py --mode dp
export CUDA_VISIBLE_DEVICES=0,1
# torchrun --nproc_per_node=2 Heat_diffusion_Cupy.py --mode ddp
python3 Heat_diffusion_Cupy.py
