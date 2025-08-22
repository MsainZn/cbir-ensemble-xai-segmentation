#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/VGG16.out
#SBATCH --error=results/VGG16.err



echo "Training Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/E/VGG16.json' \
 --pickles_path 'pickles/E' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"

echo "Training Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --config_json 'config/image/F/VGG16.json' \
 --pickles_path 'pickles/F' \
 --results_path 'results' \
 --train_or_test 'train'
echo "Finished"

