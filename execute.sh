#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=D_total
#SBATCH --output=logs/mixed_canon.%j
#SBATCH --time=0-08:00
#SBATCH --mem=32GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python experiments.py
echo 'finished'
