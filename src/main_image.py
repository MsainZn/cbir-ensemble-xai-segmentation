# Imports
import argparse
import os
import numpy as np
from datetime import datetime
import random
import json
import shutil
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# PyTorch Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

# Project Imports
from utilities_preproc import sample_manager
from utilities_traintest import TripletDataset, train_model, eval_model
from utilities_imgmodels import MODELS_DICT as models_dict, ModelEnsemble
from utilities_visualization import visualize_all_queries 
from xai_utils.xai_visualization import visualize_rankings_with_xai

# WandB Imports
import wandb

# Configuration Constants
CUDA_CONFIG = {
    'max_split_size_mb': 128,
    'garbage_collection_threshold': 0.9,
    'cuda_launch_blocking': False
}

def setup_environment(seed=10):
    """Configure reproducibility and CUDA settings"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cudnn.deterministic = True
    
    # Apply CUDA configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        f'max_split_size_mb:{CUDA_CONFIG["max_split_size_mb"]},'
        f'garbage_collection_threshold:{CUDA_CONFIG["garbage_collection_threshold"]}'
    )
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(int(CUDA_CONFIG['cuda_launch_blocking']))

def load_config(args):
    """Load configuration based on execution mode"""
    if args.train_or_test == "train":
        config_path = args.ensemble_config if args.ensemble_config else args.config_json
    else:
        config_path = os.path.join(args.checkpoint_path, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config, bool(args.ensemble_config)

def setup_paths(args, config, timestamp):
    """Create and organize output directories"""
    if args.train_or_test == "train":
        experiment_path = Path(args.results_path) / timestamp
        path_save = experiment_path / 'bin'
        experiment_path.mkdir(parents=True, exist_ok=True)
        path_save.mkdir(exist_ok=True)
        
        # Save used config
        config_src = args.ensemble_config if args.ensemble_config else args.config_json
        shutil.copyfile(config_src, experiment_path / 'config.json')
    else:
        if args.ensemble_config:
            path_save = Path(args.results_path) / 'ensemble_results' / timestamp
            path_save.mkdir(parents=True, exist_ok=True)
        else:
            path_save = Path(args.checkpoint_path) / 'bin'
    
    return path_save

def initialize_model(config, is_ensemble, device, checkpoint_path=None):
    """Initialize model with proper configuration"""
    if is_ensemble:
        model = ModelEnsemble(
            model_names=config["model_names"],
            checkpoint_paths=config.get("checkpoint_paths"),
            trainable=config.get("trainable", False),
            models_dict=models_dict,
            fusion_type=config.get("fusion_type", "projection"),
            mlp_dims=config.get("mlp_dims"),
            proj_dim=config.get("proj_dim"),
            use_l2_norm=config.get("use_l2_norm", True),
            transformer_dim=config.get("transformer_dim", 512),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 1),
            load_function=lambda x: safe_load(x, device)
        )
    else:
        model = models_dict[config["model_name"]]
        if checkpoint_path:
            model_path = checkpoint_path / "model_final.pt"
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                if any(k.startswith("base_model.") for k in state_dict.keys()):
                    state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(f"Error loading model: {e}")
    
    # Memory check before moving to device
    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(device)
        model_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        xai_buffer = 2 * 1024**3  # 2GB buffer for XAI
        
        if model_mem > (free_mem - xai_buffer):
            raise RuntimeError(f"Insufficient GPU memory. Required: {model_mem/1024**3:.2f}GB, Available: {free_mem/1024**3:.2f}GB")
    
    return model.to(device)

def create_dataloaders(QNS_list_image_train, QNS_list_image_test, 
                      QNS_list_tabular_train, QNS_list_tabular_test, 
                      transform, batch_size, use_tabular=False):
    """Create train and test dataloaders"""
    if use_tabular:
        train_dataset = TripletDataset(
            QNS_list_image=QNS_list_image_train,
            QNS_list_tabular=QNS_list_tabular_train,
            transform=transform
        )
        test_dataset = TripletDataset(
            QNS_list_image=QNS_list_image_test,
            QNS_list_tabular=QNS_list_tabular_test,
            transform=transform
        )
    else:
        train_dataset = TripletDataset(
            QNS_list=QNS_list_image_train,
            transform=transform
        )
        test_dataset = TripletDataset(
            QNS_list=QNS_list_image_test,
            transform=transform
        )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return train_loader, test_loader

def run_ensemble_diagnostics(model, config, test_loader, QNS_list_image_test, device):
    """Run diagnostic checks for ensemble models"""
    print("\n=== Running Ensemble Diagnostics ===")
    
    # Feature similarity check
    print("\n[1/2] Model Feature Similarity:")
    debug_dataset = TripletDataset(QNS_list=QNS_list_image_test[:1], 
                                 transform=model.get_transform())
    debug_loader = DataLoader(debug_dataset, batch_size=1)
    sample = next(iter(debug_loader))['query'].to(device)
    
    with torch.no_grad():
        features = [m(sample) for m in model.models]
        for i, j in combinations(range(len(features)), 2):
            sim = F.cosine_similarity(features[i], features[j]).mean().item()
            print(f"  {config['model_names'][i]} vs {config['model_names'][j]}: {sim:.4f}")

    # Individual model performance
    print("\n[2/2] Individual Model Performance:")
    for i, name in enumerate(config["model_names"]):
        try:
            single_model = models_dict[name]().to(device)
            if config["checkpoint_paths"] and i < len(config["checkpoint_paths"]):
                single_model.load_state_dict(
                    torch.load(config["checkpoint_paths"][i], map_location=device),
                    strict=False
                )
            
            _, test_acc = eval_model(single_model, test_loader, QNS_list_image_test, device)
            print(f"{name}: Test Acc = {test_acc:.4f}")
            del single_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Failed to test {name}: {str(e)}")

def safe_save_model(model, path):
    """Robust model saving with fallback to temporary location"""
    try:
        torch.save(model.state_dict(), path)
        return True
    except Exception as e:
        print(f"Failed to save model to {path}: {str(e)}")
        try:
            temp_path = f"/tmp/{os.path.basename(path)}"
            torch.save(model.state_dict(), temp_path)
            print(f"Model saved to temporary location: {temp_path}")
            return True
        except Exception as e:
            print(f"Failed to save model to temporary location: {str(e)}")
            return False

def initialize_wandb(config, timestamp):
    """Initialize Weights & Biases logging"""
    wandb_project_config = {
        "seed": config["seed"],
        "lr": config.get("lr", 0.0001),
        "num_epochs": config["num_epochs"],
        "batch_size": config["batch_size"],
        "margin": config["margin"],
        "split_ratio": config["split_ratio"],
        "catalogue_type": config["catalogue_type"],
        "doctor_code": config["doctor_code"],
        "model_name": config.get("model_name", "ensemble"),
        "fusion_type": config.get("fusion_type", "projection"),
        "transformer_dim": config.get("transformer_dim", 512),
        "nhead": config.get("nhead", 4)
    }
    return wandb.init(
        project="bcs-aesth-mm-attention-mir",
        name=config.get("model_name", "ensemble")+'_'+timestamp,
        config=wandb_project_config
    )

def main():
    parser = argparse.ArgumentParser(description='Cinderella BreLoAI Retrieval: Model Training with image data.')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--config_json', type=str, default="config/config_image.json", help="JSON config file")
    parser.add_argument('--pickles_path', type=str, required=True, help="Path to pickle files")
    parser.add_argument('--results_path', type=str, help="Path to save results")
    parser.add_argument('--train_or_test', type=str, choices=["train", "test"], default="train", 
                       help="Execution mode: train or test")
    parser.add_argument('--checkpoint_path', type=str, help="Path to model checkpoints")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--ensemble_config', type=str, help="JSON file for ensemble configuration")
    parser.add_argument('--visualize', action='store_true', help="Enable ranking visualizations")
    parser.add_argument('--visualizations_path', type=str, help="Path to save ranking visualizations")
    parser.add_argument('--visualize_all', action='store_true', 
                   help="Generate visualizations for all query images")
    parser.add_argument('--max_visualizations', type=int, default=20,
                   help="Maximum number of visualizations to generate")    
    parser.add_argument('--visualize_triplets', action='store_true', 
                   help="Generate visualizations for triplets")    
    parser.add_argument('--generate_xai', action='store_true', help='Generate explanation maps')
    parser.add_argument('--xai_backend', choices=['Captum','MONAI'], help='XAI library to use')
    parser.add_argument('--xai_method', help='Specific explanation method') 
    parser.add_argument('--xai_batch_size', type=int, default=1, help='Batch size for XAI computations')          
    parser.add_argument('--use_tabular', action='store_true', default=False, 
                   help="Whether to use tabular data along with images")
    args = parser.parse_args()

    # Setup core components
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config, is_ensemble = load_config(args)
    setup_environment(config["seed"])
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    path_save = setup_paths(args, config, timestamp)
    
    # Initialize WandB if training
    wandb_run = initialize_wandb(config, timestamp) if args.train_or_test == "train" else None

    # Load data and create model
    QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = sample_manager(
        pickles_path=args.pickles_path
    )
    
    model = initialize_model(
        config, 
        is_ensemble, 
        device,
        checkpoint_path=path_save if args.train_or_test == "test" else None
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        QNS_list_image_train,
        QNS_list_image_test,
        QNS_list_tabular_train,
        QNS_list_tabular_test,
        transform=model.get_transform(),
        batch_size=config["batch_size"],
        use_tabular=args.use_tabular
    )

    # Run diagnostics if requested
    if args.verbose and isinstance(model, ModelEnsemble):
        run_ensemble_diagnostics(model, config, test_loader, QNS_list_image_test, device)

    # Training or evaluation
    criterion = TripletMarginLoss(margin=config["margin"], p=2)
    
    if args.train_or_test == "train":
        best_model, final_epoch_loss, _ = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            QNS_list_train=QNS_list_image_train,
            QNS_list_test=QNS_list_image_test,
            optimizer=optim.Adam(model.parameters(), lr=config["lr"]),
            criterion=criterion,
            num_epochs=config["num_epochs"],
            device=device,
            path_save=path_save,
            wandb_run=wandb_run
        )
        
        if safe_save_model(best_model, path_save / "model_final.pt"):
            print("Model saved successfully")
        
        if wandb_run:
            wandb_run.finish()
    else:
        # Evaluation mode
        if args.visualize_all:
            visualize_all_queries(
                model=model,
                QNS_lists={'test': QNS_list_image_test},
                transform=model.get_transform(),
                device=device,
                output_dir=args.visualizations_path,
                max_visualizations=min(args.max_visualizations, 500),
                xai_method=args.xai_method if args.generate_xai else None,
                xai_backend=args.xai_backend if args.generate_xai else None
            )
        else:
            train_acc, train_ndcg = eval_model(
                model=model,
                eval_loader=train_loader,
                QNS_list_eval=QNS_list_image_train if args.visualize else None,
                device=device,
                visualize=args.visualize_triplets,
                output_dir=args.visualizations_path,
                xai_method=args.xai_method,
                xai_backend=args.xai_backend
            )
            
            test_acc, test_ndcg = eval_model(
                model=model,
                eval_loader=test_loader,
                QNS_list_eval=QNS_list_image_test if args.visualize else None,
                device=device,
                visualize=args.visualize_triplets,
                output_dir=args.visualizations_path,
                xai_method=args.xai_method,
                xai_backend=args.xai_backend
            )
            
            if args.generate_xai:
                visualize_rankings_with_xai(
                    model=model,
                    device=device,
                    eval_dataloader=test_loader,
                    xai_backend=args.xai_backend,
                    xai_method=args.xai_method,
                    results_dir=args.results_path,
                    batch_size=args.xai_batch_size
                )

            # Save evaluation results
            pd.DataFrame({
                "train_acc": [train_acc],
                "train_ndcg": [train_ndcg],
                "test_acc": [test_acc],
                "test_ndcg": [test_ndcg]
            }).to_csv(path_save / "eval_results.csv")

if __name__ == "__main__":
    main()