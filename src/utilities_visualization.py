import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xai_utils.xai_visualization import visualize_rankings_with_xai

def visualize_triplet(query_path, pos_path, neg_path, pos_distance, neg_distance, 
                     loss_value, correct, save_path=None):
    try:
        from PIL import Image, ImageOps, ImageDraw, ImageFont
        
        # Load and resize images
        img_height = 200
        query_img = ImageOps.contain(Image.open(query_path).convert('RGB'), (img_height, img_height))
        pos_img = ImageOps.contain(Image.open(pos_path).convert('RGB'), (img_height, img_height))
        neg_img = ImageOps.contain(Image.open(neg_path).convert('RGB'), (img_height, img_height))

        # Create canvas
        spacing = 20
        label_height = 40
        canvas_width = query_img.width + pos_img.width + neg_img.width + 2 * spacing
        canvas_height = img_height + label_height
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Font setup
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            bold_font = ImageFont.truetype("arialbd.ttf", 20)
        except:
            font = ImageFont.load_default()
            bold_font = font

        # Draw header with loss value
        header_text = f"Loss: {loss_value:.4f} | {'✓' if correct else '✗'}"
        draw.text((10, 5), header_text, font=bold_font, fill='green' if correct else 'red')

        # Draw images and distance labels
        y_offset = label_height
        x_offset = 0
        
        # Query image
        canvas.paste(query_img, (x_offset, y_offset))
        draw.text((x_offset + query_img.width//2 - 25, y_offset - 22), "QUERY", font=font, fill='black')
        x_offset += query_img.width + spacing
        
        # Positive/Negative images
        if correct:
            canvas.paste(pos_img, (x_offset, y_offset))
            draw.text((x_offset + pos_img.width//2 - 45, y_offset - 22), 
                     f"POS (d={pos_distance:.2f})", font=font, fill='green')
            x_offset += pos_img.width + spacing
            canvas.paste(neg_img, (x_offset, y_offset))
            draw.text((x_offset + neg_img.width//2 - 45, y_offset - 22), 
                     f"NEG (d={neg_distance:.2f})", font=font, fill='red')
        else:
            canvas.paste(neg_img, (x_offset, y_offset))
            draw.text((x_offset + neg_img.width//2 - 45, y_offset - 22), 
                     f"NEG (d={neg_distance:.2f})", font=font, fill='red')
            x_offset += neg_img.width + spacing
            canvas.paste(pos_img, (x_offset, y_offset))
            draw.text((x_offset + pos_img.width//2 - 45, y_offset - 22), 
                     f"POS (d={pos_distance:.2f})", font=font, fill='green')

        # Save or display
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            canvas.save(save_path)

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

def visualize_all_queries(model, QNS_lists, transform, device, output_dir, max_visualizations=5000, xai_method=None, xai_backend=None):
    """Memory-optimized visualization generator""" 
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    print(f"\nXAI Visualization Parameters:")
    print(f"- XAI Method: {xai_method}")
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
                    # Process query
                    query_tensor = transform(q_element.query_vector).unsqueeze(0)
                    if device.type == 'cuda':
                        query_tensor = query_tensor.to(device, non_blocking=True)
                    
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
                        
                        vec_i = model(neighbor_tensor)
                        if device.type == 'cuda':
                            vec_i = vec_i.cpu()
                        
                        distances.append(float(torch.norm(vec_ref - vec_i)))
                        del neighbor_tensor, vec_i
                        
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # Calculate nDCG
                    model_ordering = np.argsort(distances)
                    gt_ordering = np.arange(len(q_element.neighbor_vectors))
                    
                    # Calculate nDCG for this query
                    query_distances = [distances]
                    query_rev_orders = [[len(q_element.neighbor_vectors)-count for count in range(len(q_element.neighbor_vectors))]]
                    query_ndcg = 100 * np.mean((test_ndcg(query_distances) - test_ndcg(query_rev_orders))/(1 - test_ndcg(query_rev_orders)))
                    
                    # Generate filename with nDCG prefix
                    match = re.search(r'resized_224/(.*?)_resized_resized', q_element.query_vector)
                    query_base_name = match.group(1) if match else f"query_{i}"
                    save_path = os.path.join(set_dir, f'ndcg_{query_ndcg:.4f}_{query_base_name}.png')
                    
                    # Generate visualization
                    if xai_method:  
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
                        print(f"Generated XAI visualization for {save_path}")  # Debug output
                    else:
                        # In visualize_all_queries:
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

def visualize_rankings(query_path, neighbor_paths, model_ordering, gt_ordering, save_path=None, show_labels=True):
    """
    Generate visualization with optional labels
    
    Args:
        show_labels: If True, shows Rx-Oy labels on images (default: True)
    """
    try:
        from PIL import Image, ImageOps, ImageDraw, ImageFont

        # Load images
        query_img = Image.open(query_path).convert('RGB')
        neighbor_imgs = [Image.open(p).convert('RGB') for p in neighbor_paths]

        # Create mapping from image index to ground truth rank
        gt_rank_mapping = {}
        for rank, img_idx in enumerate(gt_ordering, 1):
            gt_rank_mapping[img_idx] = rank

        # Resize all images
        img_height = 150
        resized_query = ImageOps.contain(query_img, (img_height, img_height))
        resized_neighbors = [ImageOps.contain(img, (img_height, img_height)) for img in neighbor_imgs]

        # Split into two rows
        num_images = len(model_ordering)
        split_point = (num_images + 1) // 2
        top_row_indices = model_ordering[:split_point]
        bottom_row_indices = model_ordering[split_point:]

        def create_row(indices, start_rank):
            row_images = []
            for i, img_idx in enumerate(indices, start_rank):
                img = resized_neighbors[img_idx].copy()
                
                if show_labels:  # Only add labels if enabled
                    draw = ImageDraw.Draw(img)
                    retrieval_rank = i
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

        # Create rows (labels will be added only if show_labels=True)
        top_row_images = create_row(top_row_indices, 1)  # Starts at R1
        bottom_row_images = create_row(bottom_row_indices, len(top_row_indices)+1)  # Continues numbering
       
        # Concatenate images
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

        # Create canvas
        spacing = 20
        label_height = 30
        canvas_width = resized_query.width + spacing + max(top_row.width, bottom_row.width)
        canvas_height = label_height + top_row.height + bottom_row.height + 10
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Add query image
        query_x = 0
        query_y = label_height + (canvas_height - label_height - resized_query.height) // 2
        canvas.paste(resized_query, (query_x, query_y))
        
        if show_labels:
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
        else:
            canvas.show()

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

    def print_triplets(self):
        for i in self.triplets:
            print(i)
        
        return


    # Method: __len__
    def __len__(self):
        return len(self.triplets)


    # Method: __getitem__
    def __getitem__(self, index):
        query, pos, neg = self.triplets[index]
        return {
            'query': self.transform(query),  
            'pos': self.transform(pos),
            'neg': self.transform(neg)
        }
