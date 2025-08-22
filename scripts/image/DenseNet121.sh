#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/DenseNet.out
#SBATCH --error=results/DenseNet.err


echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/DenseNet121.json' \
  --pickles_path 'pickles/E' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/E' \
# --train_or_test 'test' \
# --checkpoint_path 'results/2025-03-31_12-37-04' \
# --verbose
#echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/DaViT_Base.json' \
#  --pickles_path 'pickles/F' \
#  --results_path 'results' \
#  --train_or_test 'train'
# echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_02-20-44/' \
# --verbose
#echo "Finished"