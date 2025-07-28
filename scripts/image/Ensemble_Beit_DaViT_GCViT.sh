#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/Ensemble.out
#SBATCH --error=results/Ensemble.err


export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_VISIBLE_DEVICES=0  
echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --ensemble_config 'config/image/E/Ensemble_Beit_DaViT_GCViT.json' \
  --pickles_path 'pickles/E' \
  --results_path 'results' \
  --train_or_test 'train' 
 echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/E' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path 'results/2025-04-14_12-19-23'
#echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/SwinV2_Base_Patch4_Window16_256.json' \
#  --pickles_path 'pickles256/F' \
#  --results_path 'results' \
#  --train_or_test 'train' 
# echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/F' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path 'results/F/2024-11-17_04-54-50/'
#echo "Finished"