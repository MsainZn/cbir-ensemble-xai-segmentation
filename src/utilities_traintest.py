# Standard library imports
import os
import re
from itertools import combinations
from io import BytesIO

# Third-party imports
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter

# Local imports
from xai_utils.utilities_xai import compute_integrated_gradients, get_xai_attribution
from xai_utils.xai_visualization import visualize_rankings_with_xai, visualize_triplet_with_xai
from utilities_visualization import visualize_triplet

# Class: TripletDataset, creating the triplets for PyTorch
class TripletDataset(Dataset):
    def __init__(self, QNS_list=None, QNS_list_image=None, QNS_list_tabular=None, transform=None):
        """
        Initializes the TripletDataset by precomputing all triplet combinations for training.
        """
        self.transform = transform
        # Select source list depending on input type
        if QNS_list is not None:
            # Image-only mode
            source_list = QNS_list
        elif QNS_list_image is not None:
            # Multimodal mode (use image data as primary)
            source_list = QNS_list_image
        else:
            raise ValueError("Must provide either QNS_list or QNS_list_image")
        
        # Pre-compute all combinations of triplets for training
        self.triplets = []
        for qns_element in source_list:
            for pair in combinations(range(qns_element.neighbor_count), 2):
                self.triplets.append((
                    qns_element.query_vector,
                    qns_element.neighbor_vectors[pair[0]],
                    qns_element.neighbor_vectors[pair[1]]
                ))

    def __len__(self):
        """Returns the number of triplets in the dataset."""
        return len(self.triplets)

    def __getitem__(self, index):
        """Retrieves a single triplet and applies the transform to each image."""
        query, pos, neg = self.triplets[index]
        return {
            'query': self.transform(query),  
            'pos': self.transform(pos),
            'neg': self.transform(neg)
        }

    def print_triplets(self):
        """Optional function that returns triplets"""
        for i in self.triplets:
            print(i)
        return

def train_model(model, train_loader, test_loader, QNS_list_train, QNS_list_test, optimizer, criterion, num_epochs, device, path_save, wandb_run):
    """
    Trains a model using triplet loss and logs metrics to Weights & Biases (WandB).
    """
    model.to(device)
    best_acc = float('-inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        batch_losses = []  # Store batch losses for visualization
        
        for data in train_loader:
            queries = data['query'].to(device)
            positives = data['pos'].to(device)
            negatives = data['neg'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - handle ensemble models
            if hasattr(model, 'models') and isinstance(model.models, list):
                # Ensemble model: average embeddings from all models
                anchor_embeddings = torch.mean(torch.stack([m(queries) for m in model.models]), dim=0)
                pos_embeddings = torch.mean(torch.stack([m(positives) for m in model.models]), dim=0)
                neg_embeddings = torch.mean(torch.stack([m(negatives) for m in model.models]), dim=0)
            else:
                # Single model
                anchor_embeddings = model(queries)
                pos_embeddings = model(positives)
                neg_embeddings = model(negatives)
            
            # Compute triplet loss
            loss = criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
            batch_losses.append(loss.item())  # Store loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * queries.size(0)
            total_samples += queries.size(0)

        # Calculate epoch loss
        epoch_loss = running_loss / total_samples

        # Evaluation
        model.eval()
        train_acc = evaluate_triplets(model, train_loader, device)
        test_acc = evaluate_triplets(model, test_loader, device)
        train_ndcg = evaluate_ndcg(QNS_list_train, model, transform=model.get_transform(), device=device)[0]
        test_ndcg = evaluate_ndcg(QNS_list_test, model, transform=model.get_transform(), device=device)[0]

        # Log metrics to WandB (if active)
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "loss": epoch_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_ndcg": train_ndcg,
                "test_ndcg": test_ndcg
            })

        # Save best model checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(path_save, f"model_best_epoch{epoch}.pt"))
            torch.save(model.state_dict(), os.path.join(path_save, "model_final.pt"))

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Train nDCG: {train_ndcg:.4f} | "
              f"Test nDCG: {test_ndcg:.4f}")

    return model, epoch_loss, best_epoch

def eval_model(model, eval_loader, QNS_list_eval, device, visualize=False, output_dir=None, max_visualizations=None, generate_xai=False, transform=None, xai_method=None, xai_backend=None):
    """
    Evaluation function with optional visualization support
    """
    model.to(device)
    model.eval()
    
    # Evaluate triplet accuracy
    eval_acc = evaluate_triplets(
        model,
        eval_loader,
        device,
        visualize=visualize,
        output_dir=output_dir,
        max_visualizations=max_visualizations,  
        generate_xai=generate_xai,              
        transform=transform,
        xai_method=xai_method,
        xai_backend=xai_backend
    )

    eval_ndcg = None
    # Compute nDCG if QNS_list_eval is provided
    if QNS_list_eval is not None:
        eval_ndcg = evaluate_ndcg(
            QNS_list_eval,
            model,
            transform=model.get_transform(),
            device=device,
            visualize=False,         
            output_dir=output_dir,
        )[0]

    return eval_acc, eval_ndcg

def evaluate_triplets(model, data_loader, device, visualize=False, output_dir=None, max_visualizations=None, generate_xai=False, transform=None, xai_method=None, xai_backend=None):
    """
    Evaluates the model on triplet data and optionally generates visualizations.
    """
    model.eval()
    total_triplets = 0
    correct_predictions = 0
    triplet_info = []
    
    # Iterate over batches
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            queries = data['query'].to(device)
            positives = data['pos'].to(device)
            negatives = data['neg'].to(device)
            
            # Forward pass - handle ensemble models
            if hasattr(model, 'models') and isinstance(model.models, list):
                # Ensemble model: average embeddings from all models
                anchor_embeddings = torch.mean(torch.stack([m(queries) for m in model.models]), dim=0)
                pos_embeddings = torch.mean(torch.stack([m(positives) for m in model.models]), dim=0)
                neg_embeddings = torch.mean(torch.stack([m(negatives) for m in model.models]), dim=0)
            else:
                # Single model
                anchor_embeddings = model(queries)
                pos_embeddings = model(positives)
                neg_embeddings = model(negatives)
            
            # Compute distances between anchor and positive/negative
            pos_distances = torch.norm(anchor_embeddings - pos_embeddings, p=2, dim=1)
            neg_distances = torch.norm(anchor_embeddings - neg_embeddings, p=2, dim=1)
            
            # Calculate per-triplet loss manually
            losses = torch.relu(pos_distances - neg_distances + 1.0)  # margin=1.0
            
            batch_correct = (pos_distances < neg_distances)
            correct_predictions += batch_correct.sum().item()
            total_triplets += queries.size(0)

            # Collect info for visualization
            if visualize and output_dir:
                for i in range(len(queries)):
                    query_path = data_loader.dataset.triplets[batch_idx * data_loader.batch_size + i][0]
                    pos_path = data_loader.dataset.triplets[batch_idx * data_loader.batch_size + i][1]
                    neg_path = data_loader.dataset.triplets[batch_idx * data_loader.batch_size + i][2]
                    
                    q_name = os.path.splitext(os.path.basename(query_path))[0]
                    p_name = os.path.splitext(os.path.basename(pos_path))[0]
                    n_name = os.path.splitext(os.path.basename(neg_path))[0]
                    
                    triplet_info.append({
                        'loss': losses[i].item(),
                        'q_path': query_path,
                        'p_path': pos_path,
                        'n_path': neg_path,
                        'pos_dist': pos_distances[i].item(),
                        'neg_dist': neg_distances[i].item(),
                        'correct': batch_correct[i].item(),
                        'q_name': q_name,
                        'p_name': p_name,
                        'n_name': n_name
                    })
    
    # Save visualizations if requested
    if visualize and output_dir and triplet_info:
        triplet_dir = os.path.join(output_dir, "triplets")
        from utilities_visualization import visualize_all_triplets
        
        visualize_all_triplets(
            triplet_info=triplet_info,
            output_dir=triplet_dir,
            max_visualizations=max_visualizations,
            generate_xai=generate_xai,
            model=model,
            device=device,
            transform=transform,
            xai_method=xai_method,
            xai_backend=xai_backend
        )

    accuracy = correct_predictions / total_triplets
    return accuracy
    
def evaluate_ndcg(QNS_list, model, transform, device, visualize=False, output_dir=None):
    """
    Evaluates the model using the nDCG metric for ranking quality.
    """
    final_order = []
    rev_orders = []
    model.eval()
    
    # Iterate over QNS_list for nDCG evaluation
    with torch.no_grad():
        for i, q_element in enumerate(QNS_list):
            # Load and transform the query image
            query_tensor = transform(q_element.query_vector).unsqueeze(0).to(device)
            
            # Handle ensemble models
            if hasattr(model, 'models') and isinstance(model.models, list):
                vec_ref = torch.mean(torch.stack([m(query_tensor) for m in model.models]), dim=0)
            else:
                vec_ref = model(query_tensor)
            
            # Get all neighbor distances
            distances = []
            for neighbor_path in q_element.neighbor_vectors:
                neighbor_tensor = transform(neighbor_path).unsqueeze(0).to(device)
                
                # Handle ensemble models
                if hasattr(model, 'models') and isinstance(model.models, list):
                    vec_i = torch.mean(torch.stack([m(neighbor_tensor) for m in model.models]), dim=0)
                else:
                    vec_i = model(neighbor_tensor)
                
                distf = torch.norm(vec_ref - vec_i)
                distances.append(distf.item())
            
            # Get model ordering (best first)
            model_ordering = np.argsort(distances)
            # Ground truth is reverse order (since original has best at end)
            gt_ordering = np.arange(len(q_element.neighbor_vectors))[::-1]
            
            final_order.append(distances)
            rev_orders.append([len(q_element.neighbor_vectors)-count for count in range(len(q_element.neighbor_vectors))])

    # Calculate nDCG accuracy
    model_acc = 100 * np.mean((test_ndcg(final_order) - test_ndcg(rev_orders))/(1 - test_ndcg(rev_orders)))
    return model_acc, final_order

def test_ndcg(distances):       
    """
    Calculates nDCG for a list of distance arrays.
    """
    res = np.zeros(len(distances))
    # Calculate nDCG for each distance list
    for i in range(len(distances)):
        dcg_aux = 0
        idcg_aux = 0
        ndcg = 0
        dist = distances[i]
        sorted_indexes = np.argsort(dist)
        new_array = np.argsort(sorted_indexes) # Contains the position of each patient in an ordered list
        for z in range(len(dist)):
            dcg_aux += (len(dist)-z) / (np.log(new_array[z]+2)/np.log(2))
            idcg_aux += (len(dist)-z) / (np.log(z+2)/np.log(2))
        res[i]= dcg_aux/idcg_aux
    return res
