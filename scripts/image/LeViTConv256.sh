#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=LeViTConv256.out
#SBATCH --error=LeViTConv256.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/E/LeViTConv256.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --train_or_test 'test' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-17_12-50-01/' \
 --verbose
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/LeViTConv256.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --train_or_test 'test' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_15-53-41/' \
 --verbose
echo "Finished"