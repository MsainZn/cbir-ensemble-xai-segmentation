#!/bin/bash
#SBATCH --partition=cpu_8cores 
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc_E.out
#SBATCH --error=dataproc_E.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Catalogue Type: E"
python src/main_dataproc.py \
 --config_json 'config/dataproc/E.json' \
 --images_resized_path 'resized_224' \
 --images_original_path 'breloai-rsz-v2' \
 --csvs_path 'csvs' \
 --pickles_path 'pickles/E'

echo "Finished"