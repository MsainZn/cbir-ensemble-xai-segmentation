#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/DaViT_Tiny.out
#SBATCH --error=results/DaViT_Tiny.err





echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/DaViT_Tiny.json' \
  --pickles_path 'pickles/E' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --config_json 'config/image/E/DaViT_Tiny.json' \
# --pickles_path 'pickles/E' \
# --train_or_test 'test' \
# --checkpoint_path 'results/E/2024-11-16_22-23-51/' \
# --verbose
#echo "Finished"

 echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/F/DaViT_Tiny.json' \
  --pickles_path 'pickles/F' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/F' \
# --train_or_test 'test' \
# --checkpoint_path 'results/F/2024-11-17_02-20-44/' \
# --verbose
#echo "Finished"