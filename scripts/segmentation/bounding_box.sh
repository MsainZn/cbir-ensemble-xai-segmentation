#!/bin/bash
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc3_E.out
#SBATCH --error=dataproc3_E.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Starting bounding box processing..."

python src/segmentation_utils/segmentation_BoundingBox.py \
  --input_folder "segmentation_post_proc" \
  --output_folder "bounding_box_output"

echo "Bounding box processing completed!"
echo "Results saved to: bounding_box_output"