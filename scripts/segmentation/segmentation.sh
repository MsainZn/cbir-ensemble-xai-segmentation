#!/bin/bash
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc3_E.out
#SBATCH --error=dataproc3_E.err


echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Starting breast segmentation pipeline..."

python src/segmentation_utils/Segmentation_application.py \
  --input_folder "breloai-rsz-v2-1024_copy" \
  --output_segmented "breloai-rsz-v2_segmented" \
  --output_debug "breloai-rsz-v2_segmented_debug" \
  --segmentation_model_path "SAM_MetaC_ce_pedro.pth"

echo "Segmentation pipeline completed successfully!"
echo "Segmented images saved to: breloai-rsz-v2_segmented"
echo "Debug masks saved to: breloai-rsz-v2_segmented_debug"
