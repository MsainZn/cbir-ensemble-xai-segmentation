#!/bin/bash
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc3_E.out
#SBATCH --error=dataproc3_E.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Starting segmentation post-processing..."

python src/segmentation_utils/SegmentationPostProcessing.py \
  --input_folder "breloai-rsz-v2_segmented_debug" \
  --output_folder "post_processed_output"

echo "Post-processing completed!"
echo "Results saved to: post_processed_output"
