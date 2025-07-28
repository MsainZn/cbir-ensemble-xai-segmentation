#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/DaViT_Base.out
#SBATCH --error=results/DaViT_Base.err

 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/DaViT_Base.json' \
  --pickles_path 'pickles/E_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"

 echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/F/DaViT_Base.json' \
  --pickles_path 'pickles/F_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"

 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/Beit_Base_Patch16_224.json' \
  --pickles_path 'pickles/E_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"

  echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/F/DaViT_Base.json' \
  --pickles_path 'pickles/F_bounding_box_bigger' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"