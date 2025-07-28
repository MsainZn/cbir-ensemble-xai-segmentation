#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=ResNeXt50_32x4d_Base_224.out
#SBATCH --error=ResNeXt50_32x4d_Base_224.err

echo "Training Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/E/ResNeXt50_32x4d_Base_224.json' \
 --pickles_path 'pickles/E_bounding_box_bigger' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"

echo "Training Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/E/ResNeXt50_32x4d_Base_224.json' \
 --pickles_path 'pickles/E_bounding_box' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"


