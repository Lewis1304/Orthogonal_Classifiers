#!/bin/bash -l
#SBATCH -n 1
#SBATCH --job-name=1sitentr
#SBATCH --output=logs/s_o_class.%j
#SBATCH --time=3-00:00
#SBATCH --mem=128GB
module load devtools/anaconda
conda deactivate
conda activate variational_orthogonal_classifiers
python experiments.py
echo 'finished'
