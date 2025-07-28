import os
import numpy as np
from PIL import Image
import argparse
from Config import COLOR_MAP, COLOR_TO_CLASS
from Visualize import image_to_mask, mask_to_image
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    label,
    binary_dilation
)

def bbox_of_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    return min_row, max_row, min_col, max_col

def masks_touch(mask1, mask2):
    dilated_mask2 = binary_dilation(mask2, structure=np.ones((3, 3)))
    return np.any(mask1 & dilated_mask2)

def process_mask(mask):
    struct = generate_binary_structure(2, 2)
    close_struct = np.ones((100, 100), dtype=bool)  # Very large closing, if used in the future, reduce size to avoid joining unwanted classes

    # Extract masks
    rb = mask == 2
    lb = mask == 3
    rn = mask == 4
    ln = mask == 5

    # Strong closing and filling
    rb_clean = binary_fill_holes(binary_closing(rb, structure=close_struct))
    lb_clean = binary_fill_holes(binary_closing(lb, structure=close_struct))
    rn_clean = binary_fill_holes(binary_closing(rn, structure=close_struct))
    ln_clean = binary_fill_holes(binary_closing(ln, structure=close_struct))

    final_mask = np.zeros_like(mask)

    # Select best breast connected to nipple
    for breast_mask, nipple_mask, breast_id in [
        (rb_clean, rn_clean, 2),
        (lb_clean, ln_clean, 3)
    ]:
        labeled, num = label(breast_mask, structure=struct)
        sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]

        valid_label = None
        for i, _ in sizes:
            comp = labeled == i
            if masks_touch(comp, nipple_mask):
                valid_label = i
                break

        if not valid_label and sizes:
            valid_label = max(sizes, key=lambda x: x[1])[0]

        if valid_label:
            final_mask[labeled == valid_label] = breast_id

    # Update cleaned breasts
    rb_final = final_mask == 2
    lb_final = final_mask == 3

    # Process nipples: keep full connected nipple components if they touch correct breast
    def preserve_valid_nipples(nipple_clean, breast_final, nipple_id):
        labeled, num = label(nipple_clean, structure=struct)
        valid_mask = np.zeros_like(nipple_clean)

        for i in range(1, num + 1):
            comp = labeled == i
            if masks_touch(breast_final, comp):
                valid_mask |= comp  # Keep entire object

        return valid_mask

    rn_valid = preserve_valid_nipples(rn_clean, rb_final, 4)
    ln_valid = preserve_valid_nipples(ln_clean, lb_final, 5)

    final_mask[rn_valid] = 4
    final_mask[ln_valid] = 5

    return final_mask


def process_segmented_image(input_path, output_folder):
    try:
        img = Image.open(input_path).convert('RGB')
        img_array = np.array(img)
        mask = image_to_mask(img_array)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_folder, exist_ok=True)

        # Save original
        original_path = os.path.join(output_folder, f"{base_name}_original.png")
        Image.fromarray(img_array).save(original_path)

        # Process and save final
        corrected_mask = process_mask(mask)
        final_image = mask_to_image(corrected_mask)
        final_image.save(os.path.join(output_folder, f"{base_name}_final.png"))

        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def process_segmented_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            in_path = os.path.join(input_folder, file)
            if process_segmented_image(in_path, output_folder):
                count += 1
    print(f"Processed {count} images. Results saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    process_segmented_folder(args.input_folder, args.output_folder)