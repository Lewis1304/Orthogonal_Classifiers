#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=sD32_s420
#SBATCH --output=logs/ortho_class.%j
#SBATCH --time=2-00:00
#SBATCH --mem=64GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python experiments.py
echo 'finished'
