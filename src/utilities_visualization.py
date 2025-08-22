import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
import re

# ========== REUSABLE HELPER FUNCTIONS ==========

def _concat_images_horizontally(images):
    """Helper function to concatenate images horizontally"""
    widths = [img.width for img in images]
    total_width = sum(widths)
    max_height = max(img.height for img in images)
    new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_img

def _create_row_with_labels(images, indices, gt_rank_mapping, start_rank=1, show_labels=True):
    """Helper function to create a row of images with labels"""
    row_images = []
    for pos, img_idx in enumerate(indices, start_rank):
        img = images[img_idx].copy()
        
        if show_labels:
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
            
            # Draw text with shadow
            draw.text((x_pos+1, 6), label, font=font, fill='black')
            draw.text((x_pos, 5), label, font=font, fill=text_color)
        
        row_images.append(img)
    return row_images

def _setup_triplet_canvas(query_img, pos_img, neg_img):
    """Helper to create canvas for triplet visualization"""
    spacing = 20
    label_height = 40
    canvas_width = query_img.width + pos_img.width + neg_img.width + 2 * spacing
    canvas_height = query_img.height + label_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    return canvas, draw, spacing, label_height

def _load_and_resize_triplet_images(query_path, pos_path, neg_path, img_height=200):
    """Helper to load and resize triplet images"""
    query_img = ImageOps.contain(Image.open(query_path).convert('RGB'), (img_height, img_height))
    pos_img = ImageOps.contain(Image.open(pos_path).convert('RGB'), (img_height, img_height))
    neg_img = ImageOps.contain(Image.open(neg_path).convert('RGB'), (img_height, img_height))
    return query_img, pos_img, neg_img

def _get_triplet_fonts():
    """Helper to get fonts for triplet visualization"""
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        bold_font = ImageFont.truetype("arialbd.ttf", 20)
    except:
        font = ImageFont.load_default()
        bold_font = font
    return font, bold_font

def _place_triplet_images(canvas, draw, query_img, pos_img, neg_img, spacing, label_height, font):
    """Helper to place images on canvas with labels"""
    y_offset = label_height
    x_offset = 0
    
    # Query image
    canvas.paste(query_img, (x_offset, y_offset))
    draw.text((x_offset + query_img.width//2 - 25, y_offset - 22), "QUERY", font=font, fill='black')
    x_offset += query_img.width + spacing
    
    # Negative image
    canvas.paste(neg_img, (x_offset, y_offset))
    draw.text((x_offset + neg_img.width//2 - 15, y_offset - 22), "NEG", font=font, fill='black')
    x_offset += pos_img.width + spacing
    
    # Positive image
    canvas.paste(pos_img, (x_offset, y_offset))
    draw.text((x_offset + pos_img.width//2 - 10, y_offset - 22), "POS", font=font, fill='black')
    
    return canvas

def _create_gt_rank_mapping(gt_ordering):
    """Create mapping from image index to ground truth rank"""
    gt_rank_mapping = {}
    for rank, img_idx in enumerate(gt_ordering, 1):
        gt_rank_mapping[img_idx] = rank
    return gt_rank_mapping

def _setup_ranking_canvas(resized_query, top_row, bottom_row, spacing=20, label_height=30):
    """Setup canvas for ranking visualization"""
    canvas_width = resized_query.width + spacing + max(top_row.width, bottom_row.width)
    canvas_height = label_height + top_row.height + bottom_row.height + 10
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    return canvas, draw, canvas_width, canvas_height

def _add_query_to_canvas(canvas, draw, resized_query, query_x, query_y, show_labels=True):
    """Add query image to canvas with optional label"""
    canvas.paste(resized_query, (query_x, query_y))
    
    if show_labels:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        label = "QUERY"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = query_x + (resized_query.width - text_width) // 2
        label_y = query_y - text_height - 5
        draw.text((label_x, label_y), label, font=font, fill='black')

def _process_query_and_neighbors(model, transform, device, q_element):
    """Process query and neighbors to get distances and ordering"""
    # Process query
    query_tensor = transform(q_element.query_vector).unsqueeze(0)
    if device.type == 'cuda':
        query_tensor = query_tensor.to(device, non_blocking=True)
    
    # Handle ensemble models
    if hasattr(model, 'models') and isinstance(model.models, list):
        vec_ref = torch.mean(torch.stack([m(query_tensor) for m in model.models]), dim=0)
    else:
        vec_ref = model(query_tensor)
        
    if device.type == 'cuda':
        vec_ref = vec_ref.cpu()
    del query_tensor
    
    # Process neighbors
    distances = []
    for neighbor_path in q_element.neighbor_vectors:
        neighbor_tensor = transform(neighbor_path).unsqueeze(0)
        if device.type == 'cuda':
            neighbor_tensor = neighbor_tensor.to(device, non_blocking=True)
        
        # Handle ensemble models
        if hasattr(model, 'models') and isinstance(model.models, list):
            vec_i = torch.mean(torch.stack([m(neighbor_tensor) for m in model.models]), dim=0)
        else:
            vec_i = model(neighbor_tensor)
            
        if device.type == 'cuda':
            vec_i = vec_i.cpu()
        
        distances.append(float(torch.norm(vec_ref - vec_i)))
        del neighbor_tensor, vec_i
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate ordering
    model_ordering = np.argsort(distances)
    gt_ordering = np.arange(len(q_element.neighbor_vectors))
    
    return distances, model_ordering, gt_ordering, vec_ref

def _calculate_query_ndcg(distances, q_element):
    from utilities_traintest import test_ndcg
    """Calculate nDCG for a query"""
    query_distances = [distances]
    query_rev_orders = [[len(q_element.neighbor_vectors)-count for count in range(len(q_element.neighbor_vectors))]]
    query_ndcg = 100 * np.mean((test_ndcg(query_distances) - test_ndcg(query_rev_orders))/(1 - test_ndcg(query_rev_orders)))
    return query_ndcg

# ========== MAIN VISUALIZATION FUNCTIONS ==========

def visualize_rankings(query_path, neighbor_paths, model_ordering, gt_ordering, save_path=None, show_labels=True):
    """
    Generate visualization with optional labels
    """
    try:
        # Load images
        query_img = Image.open(query_path).convert('RGB')
        neighbor_imgs = [Image.open(p).convert('RGB') for p in neighbor_paths]

        # Create mapping from image index to ground truth rank
        gt_rank_mapping = _create_gt_rank_mapping(gt_ordering)

        # Resize all images
        img_height = 150
        resized_query = ImageOps.contain(query_img, (img_height, img_height))
        resized_neighbors = [ImageOps.contain(img, (img_height, img_height)) for img in neighbor_imgs]

        # Split into two rows
        num_images = len(model_ordering)
        split_point = (num_images + 1) // 2
        top_row_indices = model_ordering[:split_point]
        bottom_row_indices = model_ordering[split_point:]

        # Create rows using helper function
        top_row_images = _create_row_with_labels(resized_neighbors, top_row_indices, gt_rank_mapping, 1, show_labels)
        bottom_row_images = _create_row_with_labels(resized_neighbors, bottom_row_indices, gt_rank_mapping, len(top_row_images)+1, show_labels)
       
        # Concatenate images using helper
        top_row = _concat_images_horizontally(top_row_images)
        bottom_row = _concat_images_horizontally(bottom_row_images)

        # Create canvas
        spacing = 20
        label_height = 30
        canvas, draw, canvas_width, canvas_height = _setup_ranking_canvas(resized_query, top_row, bottom_row, spacing, label_height)

        # Add query image
        query_x = 0
        query_y = label_height + (canvas_height - label_height - resized_query.height) // 2
        _add_query_to_canvas(canvas, draw, resized_query, query_x, query_y, show_labels)

        # Add results
        results_x = query_x + resized_query.width + spacing
        results_y = label_height
        canvas.paste(top_row, (results_x, results_y))
        canvas.paste(bottom_row, (results_x, results_y + top_row.height + 10))

        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)
        else:
            canvas.show()

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

def visualize_all_queries(model, QNS_lists, transform, device, output_dir, max_visualizations=5000, generate_xai=False, xai_method=None, xai_backend=None):
    from xai_utils.xai_visualization import visualize_rankings_with_xai
    """Memory-optimized visualization generator with XAI support"""
    print(f"[DEBUG] XAI generation enabled: {generate_xai}")
    if generate_xai:
        print(f"[DEBUG] XAI Method: {xai_method}")
        print(f"[DEBUG] XAI Backend: {xai_backend}")
        
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    print(f"\nVisualization Parameters:")
    print(f"- Generate XAI: {generate_xai}")
    print(f"- Device: {device}")
    print(f"- Output Dir: {output_dir}")
    print(f"- Max Visualizations: {max_visualizations}")
    
    with torch.no_grad():
        torch.set_grad_enabled(False)
        
        visualization_count = 0
        
        for set_name, QNS_list in QNS_lists.items():
            set_dir = os.path.join(output_dir, set_name)
            os.makedirs(set_dir, exist_ok=True)
            
            for i, q_element in enumerate(QNS_list):
                if visualization_count >= max_visualizations:
                    break
                
                try:
                    # Process query and neighbors
                    distances, model_ordering, gt_ordering, vec_ref = _process_query_and_neighbors(
                        model, transform, device, q_element
                    )
                    
                    # Calculate nDCG for this query
                    query_ndcg = _calculate_query_ndcg(distances, q_element)
                    
                    # Generate filename with nDCG prefix
                    query_filename = os.path.basename(q_element.query_vector)
                    save_path = os.path.join(set_dir, f'ndcg_{query_ndcg:.4f}_{query_filename}.png')

                    # Generate visualization
                    if generate_xai:  
                        visualize_rankings_with_xai(
                            q_element.query_vector,
                            q_element.neighbor_vectors,
                            model_ordering,
                            gt_ordering,
                            model=model,
                            device=device,
                            transform=transform,
                            save_path=save_path,
                            xai_method=xai_method,
                            xai_backend=xai_backend
                        )
                        print(f"Generated XAI visualization for {save_path}")
                    else:
                        visualize_rankings(
                            q_element.query_vector,
                            q_element.neighbor_vectors,
                            model_ordering,
                            gt_ordering,
                            save_path=save_path,
                            show_labels=True  
                        )
                    
                    visualization_count += 1
                    del vec_ref, distances
                    
                except Exception as e:
                    print(f"Error processing query {i}: {str(e)}")
                    continue
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    print(f"Generated {visualization_count} visualizations in {output_dir}")

def visualize_triplet(query_path, pos_path, neg_path, pos_distance, neg_distance, 
                     loss_value, correct, save_path=None):
    """Visualize triplet without XAI using helper functions"""
    try:
        # Load and resize images
        query_img, pos_img, neg_img = _load_and_resize_triplet_images(query_path, pos_path, neg_path)
        
        # Setup canvas and fonts
        canvas, draw, spacing, label_height = _setup_triplet_canvas(query_img, pos_img, neg_img)
        font, bold_font = _get_triplet_fonts()
        
        # Draw header with loss value
        #header_text = f"Loss: {loss_value:.4f} | {'✓' if correct else '✗'}"
        #draw.text((10, 5), header_text, font=bold_font, fill='green' if correct else 'red')

        # Place images with labels
        canvas = _place_triplet_images(canvas, draw, query_img, pos_img, neg_img, spacing, label_height, font)

        # Add distance labels
        draw.text((query_img.width + spacing + 10, label_height + pos_img.height + 5), 
                 f"POS d={pos_distance:.2f}", font=font, fill='green')
        draw.text((query_img.width + pos_img.width + 2*spacing + 10, label_height + neg_img.height + 5), 
                 f"NEG d={neg_distance:.2f}", font=font, fill='red')

        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)

    except Exception as e:
        print(f"Error generating triplet visualization: {str(e)}")
        
def visualize_all_triplets(triplet_info, output_dir, max_visualizations=None, generate_xai=False, model=None, device=None, transform=None, xai_method=None, xai_backend=None):
    from xai_utils.xai_visualization import visualize_triplet_with_xai
    """
    Memory-optimized triplet visualization generator with XAI support
    """
    print(f"[DEBUG] XAI generation enabled: {generate_xai}")
    if generate_xai:
        print(f"[DEBUG] XAI Method: {xai_method}")
        print(f"[DEBUG] XAI Backend: {xai_backend}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by loss (highest first)
    triplet_info.sort(key=lambda x: -x['loss'])
    
    # Determine how many triplets to save
    limit = len(triplet_info) if (max_visualizations is None or max_visualizations < 0) else min(max_visualizations, len(triplet_info))
    
    visualization_count = 0
    
    for idx, info in enumerate(triplet_info[:limit]):
        save_name = f"L-{info['loss']:.4f}_Q-{info['q_name']}_P-{info['p_name']}_N-{info['n_name']}.png"
        save_path = os.path.join(output_dir, save_name)

        try:
            if generate_xai:
                visualize_triplet_with_xai(
                    query_path=info['q_path'],
                    pos_path=info['p_path'],
                    neg_path=info['n_path'],
                    pos_distance=info['pos_dist'],
                    neg_distance=info['neg_dist'],
                    loss_value=info['loss'],
                    correct=info['correct'],
                    model=model,
                    device=device,
                    transform=transform,
                    save_path=save_path,
                    xai_method=xai_method,
                    xai_backend=xai_backend
                )
            else:
                visualize_triplet(
                    query_path=info['q_path'],
                    pos_path=info['p_path'],
                    neg_path=info['n_path'],
                    pos_distance=info['pos_dist'],
                    neg_distance=info['neg_dist'],
                    loss_value=info['loss'],
                    correct=info['correct'],
                    save_path=save_path
                )
            
            visualization_count += 1
            
        except Exception as e:
            print(f"Error visualizing triplet {idx}: {str(e)}")
            continue
    
    print(f"Generated {visualization_count} triplet visualizations in {output_dir}")