# Standard Library Imports
import argparse
import os
import sys
import time
import gc
import pickle
from datetime import datetime
from typing import List, Literal, Tuple, Optional

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast
from tqdm import tqdm

# Local Application/Library Imports
from Config import COLOR_MAP
from ConcatModels import SAM_MetaC
from Visualize import create_debug_mask
from segmentation_utils import load_sam_model, segment_and_save_image


def process_images(
    input_folder: str,
    output_segmented: str,
    output_debug: str,
    segmentation_model_path: str,
    verbose: bool = True
) -> None:
    """
    Processes all images from input_folder:
    1. Applies segmentation
    2. Saves segmented 224x224 images to output_segmented
    3. Saves debug masks to output_debug (white = breast, black = background)
    """
    # Initialize
    device = torch.device('cpu')
    os.makedirs(output_segmented, exist_ok=True)
    os.makedirs(output_debug, exist_ok=True)
    
    # Load segmentation model
    if verbose:
        print("⚙️ Loading segmentation model...")
    try:
        model = SAM_MetaC(
            in_channel_count=3,
            num_class=6,
            img_size=1024,
            is_ordinal=False
        ).to(device).eval()
        checkpoint = torch.load(segmentation_model_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        seg_model = model
        if verbose:
            print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return

    # Process each image
    image_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if verbose:
        print(f"🔍 Found {len(image_files)} images to process")
        image_files = tqdm(image_files, desc="Processing images")

    for filename in image_files:
        try:
            # Load image
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path).convert('RGB')
            
            # Verify size
            if img.size != (1024, 1024):
                if verbose:
                    print(f"⚠️ Skipping {filename} - Incorrect size: {img.size}")
                continue

            # Convert to tensor and segment
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = seg_model({'rgb': img_tensor})
                probs = torch.softmax(output, dim=1)
                mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
            
            # Create binary mask: 1 for breast (classes 2,3,4,5), 0 for background
            breast_mask = np.isin(mask, [2, 3, 4, 5]).astype(np.uint8)
            # For debug: white for breast, black for background
            debug_img = Image.fromarray((breast_mask * 255).astype(np.uint8), mode='L').convert('RGB')
            debug_path = os.path.join(output_debug, os.path.splitext(filename)[0] + '_debug.png')
            debug_img.save(debug_path)
            
            # Apply binary mask to original image (keep only breast, set background to black)
            segmented_array = np.array(img)
            segmented_array[breast_mask == 0] = [0, 0, 0]
            segmented_img = Image.fromarray(segmented_array)
            
            # Resize and save segmented image
            output_path = os.path.join(output_segmented, filename)
            segmented_img.resize((224, 224), Image.LANCZOS).save(output_path)
            
        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {str(e)}")

    if verbose:
        print(f"✅ Segmented images saved to: {output_segmented}")
        print(f"✅ Debug masks saved to: {output_debug}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_segmented', required=True)
    parser.add_argument('--output_debug', required=True)
    parser.add_argument('--segmentation_model_path', required=True)
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    process_images(
        input_folder=args.input_folder,
        output_segmented=args.output_segmented,
        output_debug=args.output_debug,
        segmentation_model_path=args.segmentation_model_path,
        verbose=args.verbose
    )