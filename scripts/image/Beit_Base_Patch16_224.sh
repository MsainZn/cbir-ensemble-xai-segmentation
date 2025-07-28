#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/Beit_Base_Patch16_224.out
#SBATCH --error=results/Beit_Base_Patch16_224.err

echo "Training Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/E/Beit_Base_Patch16_224.json' \
 --pickles_path 'pickles/E' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"

echo "Training Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/F/Beit_Base_Patch16_224.json' \
 --pickles_path 'pickles/F' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"

#echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/Beit_Base_Patch16_224.json' \
#  --pickles_path 'pickles/F' \
#  --results_path 'results' \
#  --train_or_test 'train'
#echo "Finished"

#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/F' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path 'results/2025-03-24_10-12-18'
#echo "Finished"