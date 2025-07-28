import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse
from Config import COLOR_MAP, COLOR_TO_CLASS
from Visualize import image_to_mask, mask_to_image

def get_global_bounding_box(mask, class_ids=(2, 3, 4, 5)):
    """
    Creates a global bounding box that encompasses all specified classes.
    Returns coordinates and visualization image with bounding box drawn.
    Everything inside the box is transformed to purple.
    """
    # Initialize extreme coordinates
    coords = np.argwhere(np.isin(mask, class_ids))
    
    if len(coords) == 0:
        return None, None
    
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    
    # Calculate padding - make vertical padding larger
    height = max_row - min_row
    width = max_col - min_col
    
    # Base padding (you can adjust these values)
    base_padding = 10  # pixels
    vertical_extra = 20  # extra vertical padding
    
    # Apply padding (more vertically)
    min_row = max(0, min_row - base_padding - vertical_extra)
    max_row = min(mask.shape[0] - 1, max_row + base_padding + vertical_extra)
    min_col = max(0, min_col - base_padding)
    max_col = min(mask.shape[1] - 1, max_col + base_padding)
    
    # Create visualization
    img = mask_to_image(mask)
    draw = ImageDraw.Draw(img)
    
    # Draw bounding box (white, 3px wide)
    draw.rectangle(
        [(min_col, min_row), (max_col, max_row)],
        outline="white",
        width=3
    )
    
    # Fill inside the bounding box with purple (RGB: 128, 0, 128)
    purple = (128, 0, 128)
    for y in range(min_row, max_row + 1):
        for x in range(min_col, max_col + 1):
            img.putpixel((x, y), purple)
    
    return (min_col, min_row, max_col, max_row), img

def process_image(input_path, output_path):
    """Process a single image to add bounding box"""
    try:
        img = Image.open(input_path).convert('RGB')
        img_array = np.array(img)
        mask = image_to_mask(img_array)
        
        # Get bounding box and visualization
        bbox, vis_img = get_global_bounding_box(mask)
        
        if bbox is not None:
            vis_img.save(output_path)
            return True
        return False
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """Process all images in a folder"""
    os.makedirs(output_folder, exist_ok=True)
    processed_count = 0
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if process_image(input_path, output_path):
            processed_count += 1
    
    print(f"\nSuccessfully processed {processed_count}/{len(image_files)} images")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)