#!/bin/bash -l 

#SBATCH -J GreekSBERTTraining
#SBATCH --time=02-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=expanded
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=32g 
#SBATCH --output=GreekSBERT.%j.%N.out
#SBATCH --error=GreekSBERT.%j.%N.err
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=peter.nadel@tufts.edu

echo "Starting"
date

echo "Module loading"
module load modtree/deprecated
module load anaconda/2023.07.tuftsai

echo "Activating env"
source activate general_purpose_textgen

echo "Starting SBERT Training"
python ../sbert/contrastive.py
echo "Training finished"