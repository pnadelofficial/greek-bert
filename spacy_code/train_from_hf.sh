#!/bin/bash -l 

#SBATCH -J GreekBERTTraining
#SBATCH --time=02-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --qos=expanded
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=32g 
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
source activate spacy_gpu

echo "Starting post-training with spaCy"
spacy train configs/gpu_default.cfg --paths.train corpus/train.spacy --paths.dev corpus/dev.spacy --gpu-id 0 --output ./output
echo "Post-training finished"
echo "Evaluating spaCy model"
spacy evaluate ./output/model-best corpus/test.spacy --gpu-id 0 --output ./output/eval_results.json
echo "Evaluation finished"