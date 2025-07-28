#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=ConViT_Tiny.out
#SBATCH --error=ConViT_Tiny.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/E/ConViT_Tiny.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-16_22-16-52/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-16_22-16-52/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-18_16-17-13/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/F/ConViT_Tiny.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_01-29-30/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_01-29-30/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-18_16-21-49/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"