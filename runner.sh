#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=def-npopli
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=10      # CPU cores/threads

nvidia-smi

module load python/3.11

virtualenv --no-download $SLURM_TMPDIR/code
source $SLURM_TMPDIR/code/bin/activate
pip install --no-index --upgrade pip
pip install -r ./requirements.txt
python ./adversial_and_physical_attacks.py

