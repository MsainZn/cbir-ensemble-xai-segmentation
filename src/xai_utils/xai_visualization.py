import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import traceback
import torch

# Local imports
from xai_utils.utilities_xai import get_xai_attribution, process_ig_to_heatmap, process_cam_to_heatmap, process_sbsm_to_heatmap
from utilities_visualization import (_concat_images_horizontally, _create_row_with_labels, 
                                     _setup_triplet_canvas, _load_and_resize_triplet_images, 
                                     _get_triplet_fonts, _place_triplet_images, 
                                     _create_gt_rank_mapping, _setup_ranking_canvas, 
                                     _add_query_to_canvas)

# ========== REUSABLE XAI FUNCTIONS ==========

def create_heatmap_overlay(pil_img, attribution, method='IntegratedGradients'):
    """Standardized heatmap generation with dimension handling for different XAI methods"""
    # Handle different attribution types
    if isinstance(attribution, np.ndarray):
        attr_np = attribution
    else:
        # Convert attribution to proper numpy array
        attr_np = attribution.detach().cpu().numpy()
    
    # Handle different attribution shapes
    if attr_np.ndim == 4:  # Batch dimension present
        attr_np = attr_np[0]  # Remove batch dim
    if attr_np.ndim == 3:  # Channel dimension present
        attr_np = attr_np.mean(0)  # Average across channels
    attr_np = np.squeeze(attr_np)  # Remove any remaining single-dim entries
    
    # Normalize based on method
    if method in ['IntegratedGradients', 'InputXGradient', 'GuidedGradCam']:
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        threshold = np.percentile(attr_np, 85)
    elif method in ['GradCAM', 'GradCAMpp', 'CAM']:
        attr_np = np.maximum(attr_np, 0)
        if attr_np.max() > 0:
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        threshold = np.percentile(attr_np, 70)
    elif method == 'SBSM':
        attr_np = np.maximum(attr_np, 0)
        if attr_np.max() > 0:
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        threshold = np.percentile(attr_np, 80)
    else:
        # Default processing
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

def _process_image_with_xai(image_path, model, device, transform, reference_tensor=None, method='IntegratedGradients', backend='Captum', **kwargs):
    """Process single image with XAI - supports ensemble models"""
    # Set model to training mode for gradient computation
    original_mode = model.training
    if hasattr(model, 'models') and isinstance(model.models, list):
        for m in model.models:
            m.train()
    else:
        model.train()
    
    try:
        image_tensor = transform(image_path).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        
        if reference_tensor is None:
            reference_tensor = image_tensor.clone()
        else:
            reference_tensor = reference_tensor.clone().requires_grad_(True)
        
        # Get attribution using unified interface
        attr = get_xai_attribution(
            model=model,
            in_tensor=image_tensor,
            method=method,
            backend=backend,
            reference_tensor=reference_tensor,
            **kwargs
        )
        
        # Process special methods
        if method == 'SBSM' and reference_tensor is not None:
            from xai_utils.utilities_xai import compute_sbsm
            saliency = compute_sbsm(reference_tensor, image_tensor, model)
            img = create_heatmap_overlay(Image.open(image_path).convert('RGB'), saliency, method)
        elif method in ['GradCAM', 'GradCAMpp', 'CAM'] and backend == 'MONAI':
            cam_heatmap = process_cam_to_heatmap(attr, image_tensor)
            img = create_heatmap_overlay(Image.open(image_path).convert('RGB'), cam_heatmap, method)
        else:
            img = create_heatmap_overlay(Image.open(image_path).convert('RGB'), attr, method)
        
        return img
        
    finally:
        # Restore original mode
        if hasattr(model, 'models') and isinstance(model.models, list):
            for m in model.models:
                m.train(original_mode)
        else:
            model.train(original_mode)
            
def apply_xai_to_triplet(query_path, pos_path, neg_path, model, device, transform, method='IntegratedGradients', backend='Captum', **kwargs):
    """Process triplet images with XAI - supports ensemble models"""
    # Process query
    query_img = _process_image_with_xai(query_path, model, device, transform, method=method, backend=backend, **kwargs)
    
    # Process positive and negative using query as reference
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    pos_img = _process_image_with_xai(pos_path, model, device, transform, query_tensor, method, backend, **kwargs)
    neg_img = _process_image_with_xai(neg_path, model, device, transform, query_tensor, method, backend, **kwargs)
    
    return query_img, pos_img, neg_img

def apply_xai_to_ranking(query_path, neighbor_paths, model, device, transform, method='IntegratedGradients', backend='Captum', **kwargs):
    """Process query and neighbors for ranking visualization with XAI - supports ensemble models"""
    query_img = _process_image_with_xai(query_path, model, device, transform, method=method, backend=backend, **kwargs)
    
    # Process neighbors using query as reference
    query_tensor = transform(query_path).unsqueeze(0).to(device)
    neighbor_imgs = []
    
    for path in neighbor_paths:
        neighbor_img = _process_image_with_xai(
            path, model, device, transform, query_tensor, method, backend, **kwargs
        )
        neighbor_imgs.append(neighbor_img)
    
    return query_img, neighbor_imgs

def _create_xai_ranking_visualization(query_img, neighbor_imgs, model_ordering, gt_ordering, save_path=None):
    """Create ranking visualization from XAI-processed images"""
    # Resize all images uniformly
    img_height = 150
    resized_query = ImageOps.contain(query_img, (img_height, img_height))
    resized_neighbors = [ImageOps.contain(img, (img_height, img_height)) for img in neighbor_imgs]

    # Create mapping from image index to ground truth rank
    gt_rank_mapping = _create_gt_rank_mapping(gt_ordering)

    # Split into two rows
    num_images = len(model_ordering)
    split_point = (num_images + 1) // 2
    top_row_indices = model_ordering[:split_point]
    bottom_row_indices = model_ordering[split_point:]

    # Create rows using helper function
    top_row_images = _create_row_with_labels(resized_neighbors, top_row_indices, gt_rank_mapping, 1, True)
    bottom_row_images = _create_row_with_labels(resized_neighbors, bottom_row_indices, gt_rank_mapping, len(top_row_images)+1, True)
   
    # Concatenate images using helper
    top_row = _concat_images_horizontally(top_row_images)
    bottom_row = _concat_images_horizontally(bottom_row_images)

    # Create canvas
    spacing = 20
    label_height = 30
    canvas, draw, canvas_width, canvas_height = _setup_ranking_canvas(resized_query, top_row, bottom_row, spacing, label_height)

    # Add query image (heatmapped)
    query_x = 0
    query_y = label_height + (canvas_height - label_height - resized_query.height) // 2
    canvas.paste(resized_query, (query_x, query_y))
    
    # Add query label
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    label = "QUERY"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center label horizontally above the query image
    label_x = query_x + (resized_query.width - text_width) // 2
    label_y = query_y - text_height - 5  # 5 pixels above the image

    draw.text((label_x, label_y), label, font=font, fill='black')

    # Add results
    results_x = query_x + resized_query.width + spacing
    results_y = label_height
    canvas.paste(top_row, (results_x, results_y))
    canvas.paste(bottom_row, (results_x, results_y + top_row.height + 10))

    # Save or display
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path)
        print(f"Saved XAI visualization to: {save_path}")
    else:
        canvas.show()

# ========== MAIN XAI VISUALIZATION FUNCTIONS ==========

def visualize_rankings_with_xai(query_path, neighbor_paths, model_ordering, gt_ordering, model, device, transform, save_path=None, xai_method='IntegratedGradients', xai_backend='Captum', **kwargs):
    """
    Generate XAI visualization for rankings using helper functions
    Supports ensemble models and multiple XAI methods
    """
    try:
        # Get XAI-processed images
        query_img, neighbor_imgs = apply_xai_to_ranking(
            query_path, neighbor_paths,
            model, device, transform,
            xai_method, xai_backend, **kwargs
        )
        
        # Create visualization
        _create_xai_ranking_visualization(
            query_img, neighbor_imgs, model_ordering, gt_ordering, save_path
        )

    except Exception as e:
        print(f"Error generating XAI visualization: {str(e)}")
        traceback.print_exc()

def visualize_triplet_with_xai(query_path, pos_path, neg_path, pos_distance, neg_distance, loss_value, correct, model, device, transform, save_path=None, xai_method='IntegratedGradients', xai_backend='Captum', **kwargs):
    """Triplet visualization with XAI heatmaps using helper functions - supports ensemble models"""
    try:
        # Get XAI-processed images
        query_img, pos_img, neg_img = apply_xai_to_triplet(
            query_path, pos_path, neg_path, model, device, transform,
            xai_method, xai_backend, **kwargs
        )

        # Resize images
        img_height = 200
        query_img = ImageOps.contain(query_img, (img_height, img_height))
        pos_img = ImageOps.contain(pos_img, (img_height, img_height))
        neg_img = ImageOps.contain(neg_img, (img_height, img_height))
        
        # Setup canvas and fonts 
        canvas, draw, spacing, label_height = _setup_triplet_canvas(query_img, pos_img, neg_img)
        font, bold_font = _get_triplet_fonts()
        
        # Draw header with loss value
        header_text = f"Loss: {loss_value:.4f} | {'✓' if correct else '✗'}"
        draw.text((10, 5), header_text, font=bold_font, fill='green' if correct else 'red')

        # Place images with labels 
        canvas = _place_triplet_images(canvas, draw, query_img, pos_img, neg_img, spacing, label_height, font)

        # Add distance labels
        draw.text((query_img.width + spacing + 10, label_height + pos_img.height + 5), 
                 f"POS d={pos_distance:.2f}", font=font, fill='green')
        draw.text((query_img.width + pos_img.width + 2*spacing + 10, label_height + neg_img.height + 5), 
                 f"NEG d={neg_distance:.2f}", font=font, fill='red')

        # Add XAI method info
        method_text = f"XAI: {xai_method} ({xai_backend})"
        draw.text((10, label_height + img_height + 10), method_text, font=font, fill='blue')

        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)
            print(f"Saved XAI triplet visualization to: {save_path}")

    except Exception as e:
        print(f"Error generating XAI triplet visualization: {str(e)}")
        traceback.print_exc()

def visualize_batch_xai_rankings(model, dataloader, device, transform, output_dir, xai_method='IntegratedGradients', xai_backend='Captum', max_items=50, **kwargs):
    """
    Batch XAI visualization for multiple queries - supports ensemble models
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print(f"\nBatch XAI Visualization:")
    print(f"- XAI Method: {xai_method}")
    print(f"- Backend: {xai_backend}")
    print(f"- Max Items: {max_items}")
    print(f"- Output Dir: {output_dir}")
    
    processed_count = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if processed_count >= max_items:
                break
                
            try:
                queries = data['query'].to(device)
                
                # For simplicity, process one query per batch
                for i in range(queries.size(0)):
                    if processed_count >= max_items:
                        break
                        
                    # Create dummy neighbor paths for demonstration
                    # In real usage, you'd have actual neighbor paths
                    query_path = f"query_{processed_count}"
                    neighbor_paths = [f"neighbor_{j}" for j in range(5)]  # Example
                    
                    # This would need actual image paths in real implementation
                    print(f"Processing query {processed_count} (would generate XAI with {xai_method})")
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    print(f"Processed {processed_count} items for XAI analysis")

def get_available_xai_methods(backend='Captum'):
    """Get list of available XAI methods for the specified backend"""
    if backend == 'Captum':
        return ['IntegratedGradients', 'InputXGradient', 'GuidedGradCam', 'SBSM']
    elif backend == 'MONAI':
        try:
            from monai.visualize import CAM, GradCAM, GradCAMpp, GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad
            return ['GradCAM', 'GradCAMpp', 'CAM', 'GuidedBackpropGrad', 'GuidedBackpropSmoothGrad', 'SmoothGrad', 'VanillaGrad']
        except ImportError:
            return ['GradCAM', 'GradCAMpp', 'CAM']  # Basic methods if MONAI not available
    else:
        return ['IntegratedGradients']  # Default

def validate_xai_method(method, backend='Captum'):
    """Validate if the XAI method is available for the specified backend"""
    available_methods = get_available_xai_methods(backend)
    if method not in available_methods:
        raise ValueError(f"Method '{method}' not available for backend '{backend}'. Available methods: {available_methods}")
    return True