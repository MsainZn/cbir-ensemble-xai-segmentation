#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb_ext
#SBATCH --exclude=01.ctm-deep-06
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=results/GC_ViT.out
#SBATCH --error=results/GC_ViT.err

echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/GC_ViT_224.json' \
  --pickles_path 'pickles/E_segmented_post_proc' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"


echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/F/GC_ViT_224.json' \
  --pickles_path 'pickles/F_segmented_post_proc' \
  --results_path 'results' \
  --train_or_test 'train'
 echo "Finished"


