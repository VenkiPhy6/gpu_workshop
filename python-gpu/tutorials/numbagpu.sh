#!/bin/bash
#SBATCH --job-name=Handson1
#SBATCH --reservation=hpcre-gpu
#SBATCH --time=00:30:00
#SBATCH --mem=1GB
#SBATCH -w g-04-02
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --partition=h100

module purge
module load gnu12
module load python/3.11.11
module load cuda/12.6
source venv/bin/activate
#pip3 install numba
python3 Heat_diffusion_Numba_GPU.py
