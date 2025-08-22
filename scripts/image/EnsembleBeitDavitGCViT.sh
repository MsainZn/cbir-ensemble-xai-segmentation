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
  --pickles_path 'pickles/E_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train' 
 echo "Finished"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_VISIBLE_DEVICES=0  
echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --ensemble_config 'config/image/F/Ensemble_Beit_DaViT_GCViT.json' \
  --pickles_path 'pickles/F_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train' 
 echo "Finished"


#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
#export CUDA_VISIBLE_DEVICES=0  
#echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Test Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --pickles_path 'pickles/F_bounding_box_bigger' \
#  --verbose \
#  --train_or_test 'test' \
#  --checkpoint_path 'results/2025-08-14_8-36-43'
# echo "Finished"

#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
#export CUDA_VISIBLE_DEVICES=0  
#echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Test Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --pickles_path 'pickles/F_bounding_box_bigger' \
#  --verbose \
#  --train_or_test 'test' \
#  --checkpoint_path 'results/2025-08-14_10-36-43'
# echo "Finished"
