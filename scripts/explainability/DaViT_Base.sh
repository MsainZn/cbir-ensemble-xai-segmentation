#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/DaViT_Base.out
#SBATCH --error=results/DaViT_Base.err


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
 --checkpoint_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/2025-08-06_19-38-58' \
 --xai_backend 'Captum' \
 --xai_method 'IntegratedGradients' \
 --xai_batch_size 1
echo "Finished"
