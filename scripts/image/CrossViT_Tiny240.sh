#!/bin/bash
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=CrossViT_Tiny240.out
#SBATCH --error=CrossViT_Tiny240.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/E/CrossViT_Tiny240.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --train_or_test 'test' \
 --verbose \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-16_22-19-53/'
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/CrossViT_Tiny240.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F' \
#  --train_or_test 'train'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_image.py \
 --gpu_id 0 \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --train_or_test 'test' \
 --verbose \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_01-29-49/'
echo "Finished"