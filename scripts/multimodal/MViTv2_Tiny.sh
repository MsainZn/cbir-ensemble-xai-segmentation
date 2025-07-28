#!/bin/bash
#SBATCH --partition=gpu_min80gb
#SBATCH --qos=gpu_min80gb
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=MViTv2_Tiny.out
#SBATCH --error=MViTv2_Tiny.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/E/MViTv2_Tiny.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-17_12-51-00/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E'
# echo "Finished"
echo "Testing Catalogue Type: E"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-17_12-51-00/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-18_08-58-38/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_multimodal.py \
#  --gpu_id 0 \
#  --config_json 'config/multimodal/F/MViTv2_Tiny.json' \
#  --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_17-06-22/bin/model_final.pt' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F'
# echo "Finished"
echo "Testing Catalogue Type: F"
python src/main_multimodal.py \
 --gpu_id 0 \
 --csvs_path '/nas-ctm01/datasets/private/CINDERELLA/breloai-web-db/csvs' \
 --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
 --img_model_weights_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_17-06-22/bin/model_final.pt' \
 --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-18_09-03-49/' \
 --train_or_test 'test' \
 --verbose
echo "Finished"