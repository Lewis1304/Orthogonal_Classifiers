#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=t1s_tortho
#SBATCH --output=logs/ortho_class.%j
#SBATCH --time=2-00:00
#SBATCH --mem=24GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python experiments.py
echo 'finished'
