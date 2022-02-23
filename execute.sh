#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=D_encode_preds
#SBATCH --output=logs/mixed_canon.%j
#SBATCH --time=0-12:00
#SBATCH --mem=16GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python paper_experiments.py
echo 'finished'
