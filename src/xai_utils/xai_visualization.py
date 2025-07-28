import numpy as np
from PIL import Image
from xai_utils.utilities_xai import get_xai_attribution


def create_heatmap_overlay(pil_img, attribution):
    """Standardized heatmap generation with dimension handling"""
    # Convert attribution to proper numpy array
    attr_np = attribution.detach().cpu().numpy()
    
    # Handle different attribution shapes
    if attr_np.ndim == 4:  # Batch dimension present
        attr_np = attr_np[0]  # Remove batch dim
    if attr_np.ndim == 3:  # Channel dimension present
        attr_np = attr_np.mean(0)  # Average across channels
    attr_np = np.squeeze(attr_np)  # Remove any remaining single-dim entries
    
    # Normalize and threshold
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    threshold = np.percentile(attr_np, 85)
    
    # Create overlay
    green = np.zeros((*attr_np.shape, 4))  # RGBA array
    green[..., :3] = [0.043, 0.859, 0.082]  # RGB for #0bdb15
    green[..., 3] = np.where(attr_np >= threshold, 0.7, 0)  # Alpha channel
    
    # Convert to PIL Image
    overlay_img = Image.fromarray((green * 255).astype(np.uint8), 'RGBA')
    base_img = pil_img.convert('RGBA')
    return Image.alpha_composite(base_img, overlay_img).convert('RGB')

def apply_xai_to_triplet(query_path, pos_path, neg_path, model, device, transform, method, backend):
    """Process all three triplet images with consistent XAI"""
    # Process query
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    query_attr = get_xai_attribution(model, query_tensor, method, backend)
    query_img = create_heatmap_overlay(Image.open(query_path).convert('RGB'), query_attr)
    
    # Process positive
    pos_tensor = transform(pos_path).unsqueeze(0).to(device)
    pos_attr = get_xai_attribution(model, pos_tensor, method, backend, reference=query_tensor)
    pos_img = create_heatmap_overlay(Image.open(pos_path).convert('RGB'), pos_attr)
    
    # Process negative
    neg_tensor = transform(neg_path).unsqueeze(0).to(device)
    neg_attr = get_xai_attribution(model, neg_tensor, method, backend, reference=query_tensor)
    neg_img = create_heatmap_overlay(Image.open(neg_path).convert('RGB'), neg_attr)
    
    return query_img, pos_img, neg_img

def apply_xai_to_ranking(query_path, neighbor_paths, model, device, transform, method, backend):
    """Process query and neighbors for ranking visualization"""
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    query_attr = get_xai_attribution(model, query_tensor, method, backend)
    query_img = create_heatmap_overlay(
        Image.open(query_path).convert('RGB'),
        query_attr
    )
    
    neighbor_imgs = []
    for path in neighbor_paths:
        neighbor_tensor = transform(path).unsqueeze(0).to(device)
        neighbor_attr = get_xai_attribution(
            model, 
            neighbor_tensor, 
            method, 
            backend,
            reference_tensor=query_tensor  # Pass reference for IG
        )
        neighbor_imgs.append(create_heatmap_overlay(
            Image.open(path).convert('RGB'),
            neighbor_attr
        ))
    
    return query_img, neighbor_imgs
    
def visualize_rankings_with_xai(
    query_path, 
    neighbor_paths, 
    model_ordering, 
    gt_ordering, 
    model, 
    device, 
    transform, 
    save_path=None, 
    xai_method='IntegratedGradients',
    xai_backend='Captum'
):
    try:
        from PIL import ImageOps, ImageDraw, ImageFont
        import os
        import traceback
        
        # Get XAI-processed images
        query_img, neighbor_imgs = apply_xai_to_ranking(
            query_path, neighbor_paths,
            model, device, transform,
            xai_method, xai_backend
        )
        
        # Resize all images uniformly
        img_height = 150
        resized_query = ImageOps.contain(query_img, (img_height, img_height))
        resized_neighbors = [ImageOps.contain(img, (img_height, img_height)) for img in neighbor_imgs]

        # Split into two rows (half top, half bottom)
        num_images = len(model_ordering)
        split_point = (num_images + 1) // 2
        top_row_indices = model_ordering[:split_point]
        bottom_row_indices = model_ordering[split_point:]

        # Add labels (RxOy) to heatmapped images
        def create_row(indices, start_rank=1):
            row_images = []
            gt_rank_mapping = {img_idx: rank for rank, img_idx in enumerate(gt_ordering, 1)}
            for pos, img_idx in enumerate(indices, start_rank):
                img = resized_neighbors[img_idx].copy()
                draw = ImageDraw.Draw(img)
                retrieval_rank = pos
                gt_rank = gt_rank_mapping[img_idx]
                label = f"R{retrieval_rank}-O{gt_rank}"
                text_color = 'green' if retrieval_rank == gt_rank else 'red'
                try:
                    font = ImageFont.truetype("arialbd.ttf", 20)
                except:
                    font = ImageFont.load_default()
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                x_pos = (img.width - text_width) // 2
                draw.text((x_pos+1, 6), label, font=font, fill='black')  # Shadow
                draw.text((x_pos, 5), label, font=font, fill=text_color)
                row_images.append(img)
            return row_images

        # Create both rows
        top_row_images = create_row(top_row_indices, 1)
        bottom_row_images = create_row(bottom_row_indices, len(top_row_images) + 1)

        # Concatenate images horizontally
        def concat_images(images):
            widths = [img.width for img in images]
            total_width = sum(widths)
            max_height = max(img.height for img in images)
            new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for img in images:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width
            return new_img

        top_row = concat_images(top_row_images)
        bottom_row = concat_images(bottom_row_images)

        # Create final canvas
        spacing = 20
        label_height = 30
        canvas_width = resized_query.width + spacing + max(top_row.width, bottom_row.width)
        canvas_height = label_height + top_row.height + bottom_row.height + 10
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Add query image (heatmapped)
        query_x = 0
        query_y = label_height + (canvas_height - label_height - resized_query.height) // 2
        canvas.paste(resized_query, (query_x, query_y))
        
        # Add query label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((query_x + resized_query.width // 2 - 25, 5), "QUERY", font=font, fill='black')

        # Add results
        results_x = query_x + resized_query.width + spacing
        results_y = label_height
        canvas.paste(top_row, (results_x, results_y))
        canvas.paste(bottom_row, (results_x, results_y + top_row.height + 10))

        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)
            print(f"Saved visualization to: {save_path}")
        else:
            canvas.show()

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
