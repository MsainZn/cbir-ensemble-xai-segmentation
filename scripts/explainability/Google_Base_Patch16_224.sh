#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/Google_Base_Patch16_224.out
#SBATCH --error=results/Google_Base_Patch16_224.err


VIS_DIR="visualizations/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p $VIS_DIR 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=0  

python src/main_image.py \
 --visualizations_path "$VIS_DIR" \
 --gpu_id 0 \
 --pickles_path 'pickles/F' \
 --verbose \
 --train_or_test 'test' \
 --visualize_triplets \
 --generate_xai \
 --max_visualizations 10 \
 --results_path 'results' \
 --checkpoint_path 'results/2025-06-19_06-57-17,GoogleViT_E' \
 --xai_backend 'Captum' \
 --xai_method 'IntegratedGradients' \
 --xai_batch_size 1
echo "Finished"
