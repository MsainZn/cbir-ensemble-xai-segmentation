#!/bin/bash
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc3_E.out
#SBATCH --error=dataproc3_E.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Starting mask overlap processing..."

python src/segmentation_utils/segmentation_overlap.py \
  --new_original_folder "breloai-rsz-v2-1024_copy" \
  --mask_folder "preproc_defected_img_corrected" \
  --output_folder "correct_defected"

echo "Mask overlap processing completed!"
echo "Final images saved to: correct_defected"
