#!/bin/bash -l 

#SBATCH -J GreekBERTTraining
#SBATCH --time=04-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:8
#SBATCH --qos=expanded
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=128g 
#SBATCH --output=GreekBERT.%j.%N.out
#SBATCH --error=GreekBERT.%j.%N.err
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=peter.nadel@tufts.edu

echo "Starting"
date

echo "Module loading"
module load modtree/deprecated
module load anaconda/2023.07.tuftsai

echo "Activating env"
source activate torchrun

echo "Starting BERT Training"
NUM_GPUS=${1:-8}
torchrun --standalone --nnodes 1 --nproc_per_node=$NUM_GPUS train.py --config-path=train_config.yaml

echo "Training finished"