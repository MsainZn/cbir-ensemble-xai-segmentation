#!/bin/bash
#SBATCH --partition=debug_8gb
#SBATCH --qos=debug_8gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/SwinV2.out
#SBATCH --error=results/SwinV2.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/E/SwinV2_Base_Patch4_Window16_256.json' \
#  --pickles_path 'pickles256/E' \
#  --results_path 'results' \
#  --train_or_test 'train' 
# echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/SwinV2_Base_Patch4_Window16_256.json' \
#  --pickles_path 'pickles256/F' \
#  --results_path 'results' \
#  --train_or_test 'train' 
# echo "Finished"

#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path 'pickles/E' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path 'results/2025-04-07_00-28-55'
#echo "Finished"

echo "Testing Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path 'pickles256/F' \
 --verbose \
 --train_or_test 'test' \
 --checkpoint_path 'results/2025-04-07_22-35-02,SwinV2'
echo "Finished"

