# ImportsMore actions
import argparse
import os
import numpy as np
import random
import json

# PyTorch Imports
import torch

# Project Imports
from utilities_preproc import data_preprocessing


# Seed for reproducibility purposes
def set_seed(seed=10):
    # Random Seed
    random.seed(seed)
    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy Seed
    np.random.seed(seed)
    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='CINDERELLA BreLoAI Retrieval: Data preprocessing.')
    parser.add_argument('--config_json', type=str, default="config/config_image.json", help="The JSON configuration file.")
    parser.add_argument('--images_resized_path', type=str, required=True, help="The path to the resized images.")
    parser.add_argument('--images_original_path', type=str, required=True, help="The path to the original images.")
    parser.add_argument('--csvs_path', type=str, required=True, help="The path to the CSVs with metadata.")
    parser.add_argument('--pickles_path', type=str, required=True, help="The path to the pickle files (to speed up training).")
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose.")
    args = parser.parse_args()

    # Get arguments
    config_json_ = args.config_json
    images_resized_path = args.images_resized_path
    images_original_path = args.images_original_path
    csvs_path = args.csvs_path
    pickles_path = args.pickles_path
    verbose = args.verbose

    # Build paths
    favorite_image_info = os.path.join(csvs_path, "favorite_image_info.csv")
    patient_info = os.path.join(csvs_path, "patient_info.csv")
    patient_images_info = os.path.join(csvs_path, "patient_images.csv")
    catalogue_info = os.path.join(csvs_path, "catalogue_info.csv")
    catalogue_user_info = os.path.join(csvs_path, "catalogue_user_info.csv")

    for path in [images_resized_path, pickles_path]:
        os.makedirs(path, exist_ok=True)

    # Open configuration JSON
    with open(config_json_, 'r') as j:
        config_json = json.load(j)

    # Set seed(s)
    set_seed(seed=config_json["seed"])


    # Preprocessing
    data_preprocessing(
        images_resized_path=images_resized_path,
        images_original_path=images_original_path,
        pickles_path=pickles_path,
        catalogue_info=catalogue_info,
        catalogue_user_info=catalogue_user_info,
        patient_info=patient_info,
        favorite_image_info=favorite_image_info,
        patient_images_info=patient_images_info,
        catalogue_type=config_json["catalogue_type"],
        doctor_code=config_json["doctor_code"],
        split_ratio=config_json["split_ratio"],
        seed=config_json["seed"]
    )