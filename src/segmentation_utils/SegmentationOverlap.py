import os
from PIL import Image
import numpy as np
import argparse

def find_matching_files(new_original_folder, mask_folder):
    """
    Match files from new original images (1024x1024) and mask folder by base name (ignore extension, case sensitive).
    Returns list of (original_path, mask_path, base_name).
    """
    file_pairs = []

    # Map base name (case sensitive) to original image path
    original_map = {}
    for fname in os.listdir(new_original_folder):
        if fname.lower().endswith(('.jpg', '.jpeg')):
            base = os.path.splitext(fname)[0]
            original_map[base] = os.path.join(new_original_folder, fname)

    # Map base name (case sensitive) to mask image path
    mask_map = {}
    for fname in os.listdir(mask_folder):
        if fname.lower().endswith('.png'):
            base = os.path.splitext(fname)[0]
            mask_map[base] = os.path.join(mask_folder, fname)

    # Find matches by base name (case sensitive)
    for base in original_map:
        if base in mask_map:
            file_pairs.append((original_map[base], mask_map[base], base))

    return file_pairs

def apply_mask(original_image, mask_image):
    """
    Apply the mask to the original image, keeping only the white (255,255,255) parts of the mask.
    All other pixels are set to black.
    """
    original_arr = np.array(original_image)
    mask_arr = np.array(mask_image)

    # Create a binary mask where mask is exactly white (255,255,255)
    binary_mask = np.all(mask_arr == [255, 255, 255], axis=-1)

    masked_image = np.zeros_like(original_arr)
    masked_image[binary_mask] = original_arr[binary_mask]

    return Image.fromarray(masked_image)

def process_images(new_original_folder, mask_folder, output_folder):
    """
    Process all matching image pairs from new original images and masks.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_pairs = find_matching_files(new_original_folder, mask_folder)

    for original_path, mask_path, base_name in file_pairs:
        try:
            original_img = Image.open(original_path).convert('RGB')
            mask_img = Image.open(mask_path).convert('RGB')

            if original_img.size != mask_img.size:
                mask_img = mask_img.resize(original_img.size)

            result_img = apply_mask(original_img, mask_img)

            output_path = os.path.join(output_folder, f"{base_name}.jpg")
            result_img.save(output_path)
            print(f"Processed: {base_name}")

        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_original_folder', required=True)
    parser.add_argument('--mask_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    process_images(
        new_original_folder=args.new_original_folder,
        mask_folder=args.mask_folder,
        output_folder=args.output_folder
    )