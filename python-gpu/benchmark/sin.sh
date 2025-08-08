#!/bin/bash
#SBATCH --job-name=Sin_NUMPY_CUPY
#SBATCH --reservation=hpcre-gpu
#SBATCH --time=00:30:00
#SBATCH --mem=1GB
#SBATCH -w g-04-02
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --partition=h100

module purge
module load gnu12
module load cuda/12.6
module load python/3.11.11
python3 -m venv newenv
source newenv/bin/activate
pip3 install cupy-cuda12x
python3 sin.py
