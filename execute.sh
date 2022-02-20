#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=D_encode
#SBATCH --output=logs/mixed_canon.%j
#SBATCH --time=1-00:00
#SBATCH --mem=64GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python experiments.py
echo 'finished'
