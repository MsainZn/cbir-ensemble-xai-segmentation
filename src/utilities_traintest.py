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
from xai_utils.utilities_xai import compute_attributions, compute_integrated_gradients
from utilities_imgmodels import ModelEnsemble  
from xai_utils.xai_visualization import visualize_rankings_with_xai
from utilities_visualization import visualize_triplet

# Class: TripletDataset, creating the triplets for PyTorch
class TripletDataset(Dataset):

    # Method: __init__
    def __init__(self, QNS_list, transform):

        # Class variables
        self.transform = transform

        # Pre-compute all combination of the triplets
        self.triplets = []
        for qns_element in QNS_list:
            for pair in combinations(range(qns_element.neighbor_count), 2):
                self.triplets.append(
                    (
                        qns_element.query_vector,
                        qns_element.neighbor_vectors[pair[0]],
                        qns_element.neighbor_vectors[pair[1]]
                    )
                )

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

def train_model(model, train_loader, test_loader, QNS_list_train, QNS_list_test, optimizer, criterion, num_epochs, device, path_save, wandb_run):
    """
    Trains a model using triplet loss and logs metrics to Weights & Biases (WandB).

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        QNS_list_train: List of training QNS objects.
        QNS_list_test: List of test QNS objects.
        optimizer: Optimizer for training.
        criterion: Loss function (e.g., TripletMarginLoss).
        num_epochs: Number of training epochs.
        device: Device (CPU/GPU).
        path_save: Path to save model checkpoints.
        wandb_run: Active WandB run object (for logging).
    
    Returns:
        model: Trained model.
        final_epoch_loss: Loss from the last epoch.
        best_epoch: Best epoch based on test accuracy.
    """
    model.to(device)
    best_acc = float('-inf')
    best_epoch = 0

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
            
            # Forward pass
            anchor_embeddings = model(queries)
            pos_embeddings = model(positives)
            neg_embeddings = model(negatives)
            
            # Compute loss
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

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            if isinstance(model, ModelEnsemble):
                # Save each component separately
                for i, component in enumerate(model.models):
                    torch.save(component.state_dict(), os.path.join(path_save, f"component_{i}_best_epoch{epoch}.pt"))
                    torch.save(component.state_dict(), os.path.join(path_save, f"component_{i}_final.pt"))
            else:
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
            
def eval_model(model, eval_loader, QNS_list_eval, device, visualize=False, output_dir=None, 
               xai_method=None, xai_backend=None):
    """Evaluation function with optional visualization support
    
    Args:
        model: Model to evaluate
        eval_loader: DataLoader for evaluation
        QNS_list_eval: List of query-neighbor sets for nDCG calculation
        device: Device to run evaluation on
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations
        xai_method: XAI method to use (if None, no XAI)
        xai_backend: XAI backend to use ('Captum' or 'MONAI')
    """
    model.to(device)
    model.eval()

    eval_acc = evaluate_triplets(model, eval_loader, device, visualize=visualize, output_dir=output_dir)
    
    # Only calculate nDCG and visualize if requested
    eval_ndcg = None
    if visualize or QNS_list_eval is not None:
        eval_ndcg = evaluate_ndcg(
            QNS_list_eval, 
            model, 
            transform=model.get_transform(), 
            device=device,
            visualize=visualize,
            output_dir=output_dir,
            xai_method=xai_method,  
            xai_backend=xai_backend
        )[0]

    return eval_acc, eval_ndcg
    
def evaluate_triplets(model, data_loader, device, visualize=False, output_dir=None):
    model.eval()
    total_triplets = 0
    correct_predictions = 0
    triplet_info = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            queries = data['query'].to(device)
            positives = data['pos'].to(device)
            negatives = data['neg'].to(device)
            
            anchor_embeddings = model(queries)
            pos_embeddings = model(positives)
            neg_embeddings = model(negatives)
            
            # Compute distances
            pos_distances = torch.norm(anchor_embeddings - pos_embeddings, p=2, dim=1)
            neg_distances = torch.norm(anchor_embeddings - neg_embeddings, p=2, dim=1)
            
            # Calculate per-triplet loss manually
            losses = torch.relu(pos_distances - neg_distances + 1.0)  # margin=1.0
            
            batch_correct = (pos_distances < neg_distances)
            correct_predictions += batch_correct.sum().item()
            total_triplets += queries.size(0)

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

    # Save visualizations sorted by loss (highest first)
    if visualize and output_dir and triplet_info:
        triplet_dir = os.path.join(output_dir, "triplets")
        os.makedirs(triplet_dir, exist_ok=True)
        triplet_info.sort(key=lambda x: -x['loss'])
        
        for idx, info in enumerate(triplet_info):
            save_name = f"L-{info['loss']:.4f}_Q-{info['q_name']}_P-{info['p_name']}_N-{info['n_name']}.png"
            save_path = os.path.join(triplet_dir, save_name)
            
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

    accuracy = correct_predictions / total_triplets
    return accuracy

# Function: Evaluate models using nDCG metric adjusted
def evaluate_ndcg(QNS_list, model, transform, device, visualize=False, output_dir=None, xai_method='IntegratedGradients', xai_backend='Captum'):
    final_order = []
    rev_orders = []
    model.eval()
    
    with torch.no_grad():
        for i, q_element in enumerate(QNS_list):
            fss = []
            rss = []

            # Load and transform the query image
            query_tensor = transform(q_element.query_vector).unsqueeze(0).to(device)
            vec_ref = model(query_tensor)
            
            # Get all neighbor distances
            distances = []
            for neighbor_path in q_element.neighbor_vectors:
                neighbor_tensor = transform(neighbor_path).unsqueeze(0).to(device)
                vec_i = model(neighbor_tensor)
                distf = torch.norm(vec_ref - vec_i)
                distances.append(distf.item())
            
            # Get model ordering (best first)
            model_ordering = np.argsort(distances)
            # Ground truth is reverse order (since original has best at end)
            gt_ordering = np.arange(len(q_element.neighbor_vectors))[::-1]
            
            final_order.append(distances)
            rev_orders.append([len(q_element.neighbor_vectors)-count for count in range(len(q_element.neighbor_vectors))])
            
            # Visualize if requested
            if visualize and output_dir:
                save_path = os.path.join(output_dir, f'ranking_{i}.png')
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

    model_acc = 100 * np.mean((test_ndcg(final_order) - test_ndcg(rev_orders))/(1 - test_ndcg(rev_orders)))

    return model_acc, final_order

# Function: Calculate nDCG using sorted distances
def test_ndcg(distances):       
  res = np.zeros(len(distances))
  for i in range(len(distances)):
    dcg_aux = 0
    idcg_aux = 0
    ndcg = 0
    dist = distances[i]
    sorted_indexes = np.argsort(dist)
    new_array = np.argsort(sorted_indexes) #Contains the position of each patient in an ordered list
    for z in range(len(dist)):      
      dcg_aux += (len(dist)-z) / (np.log(new_array[z]+2)/np.log(2))
      idcg_aux += (len(dist)-z) / (np.log(z+2)/np.log(2))

    res[i]= dcg_aux/idcg_aux

  return res


    