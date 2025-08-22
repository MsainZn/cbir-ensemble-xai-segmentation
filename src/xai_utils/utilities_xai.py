# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

# Captum Imports
from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    GuidedGradCam,
)

# MONAI Imports (conditional)
try:
    from monai.visualize import CAM, GradCAM, GradCAMpp, GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

def compute_integrated_gradients(model, query_tensor, neighbor_tensor, baseline=None, steps=50):
    """
    Compute IG attributions using L2 distance (consistent with retrieval).
    """
    if baseline is None:
        baseline = 0 * neighbor_tensor  # Black image baseline
    
    # Ensure tensors require gradients
    neighbor_tensor = neighbor_tensor.clone().requires_grad_(True)
    query_tensor = query_tensor.clone().requires_grad_(True)
    
    def similarity_fn(neighbor_input):
        # Ensure we're in training mode for gradient computation
        original_mode = model.training
        model.train()
        
        try:
            # Handle ensemble models
            if hasattr(model, 'models') and isinstance(model.models, list):
                # Ensure each model is in training mode
                for m in model.models:
                    m.train()
                query_embed = torch.mean(torch.stack([m(query_tensor) for m in model.models]), dim=0)
            else:
                model.train()
                query_embed = model(query_tensor)
                
            # Handle ensemble models for neighbor
            if hasattr(model, 'models') and isinstance(model.models, list):
                neighbor_embed = torch.mean(torch.stack([m(neighbor_input) for m in model.models]), dim=0)
            else:
                neighbor_embed = model(neighbor_input)
                
            # Compute similarity (negative distance for maximization)
            similarity = -torch.norm(query_embed - neighbor_embed, p=2)
            
            return similarity.unsqueeze(0)
            
        finally:
            # Restore original mode
            if hasattr(model, 'models') and isinstance(model.models, list):
                for m in model.models:
                    m.train(original_mode)
            else:
                model.train(original_mode)
    
    ig = IntegratedGradients(similarity_fn)
    
    # Stable attribution computation
    attributions = ig.attribute(
        inputs=neighbor_tensor,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
        return_convergence_delta=False
    )
    
    return attributions
    
def compute_input_x_gradient(model, in_tensor, target):
    """Compute Input X Gradient attributions"""
    in_tensor.requires_grad_()
    
    def similarity_fn(input_tensor):
        # Handle ensemble models
        if hasattr(model, 'models') and isinstance(model.models, list):
            output = torch.mean(torch.stack([m(input_tensor) for m in model.models]), dim=0)
        else:
            output = model(input_tensor)
        return output[:, target]  # For classification-style methods
    
    ixg = InputXGradient(similarity_fn)
    attribution = ixg.attribute(inputs=in_tensor, target=target)
    return attribution

def compute_guided_grad_cam(model, model_last_conv_layer, in_tensor, target):
    """Compute Guided GradCAM attributions"""
    in_tensor.requires_grad_()
    
    def similarity_fn(input_tensor):
        # Handle ensemble models - use first model for Guided GradCAM
        if hasattr(model, 'models') and isinstance(model.models, list):
            output = model.models[0](input_tensor)
        else:
            output = model(input_tensor)
        return output[:, target]
    
    ggc = GuidedGradCam(similarity_fn, model_last_conv_layer)
    attribution = ggc.attribute(inputs=in_tensor, target=target)
    return attribution

def compute_attributions(model, in_tensor, target, method, **kwargs):
    """Compute attributions using specified method"""
    device = next(model.parameters()).device
    in_tensor = in_tensor.to(device)
    
    assert method in (
        'IntegratedGradients',
        'InputXGradient',
        'GuidedGradCam',
    ), 'Please provide a valid Captum method.'

    # Select attribution method
    if method == 'IntegratedGradients':
        attribution = compute_integrated_gradients(model, in_tensor, in_tensor)
    elif method == 'InputXGradient':
        attribution = compute_input_x_gradient(model, in_tensor, target)
    elif method == 'GuidedGradCam':
        if 'model_last_conv_layer' not in kwargs:
            raise ValueError("GuidedGradCam requires model_last_conv_layer parameter")
        attribution = compute_guided_grad_cam(model, kwargs['model_last_conv_layer'], in_tensor, target)
    
    return attribution.to(device)

def compute_monai_results(in_tensor, class_idx, method, model):
    """Compute attributions using MONAI methods"""
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is not available. Please install it or use Captum methods.")
    
    # Handle ensemble models - use first model for MONAI
    if hasattr(model, 'models') and isinstance(model.models, list):
        model_to_use = model.models[0]
    else:
        model_to_use = model
    
    if method == 'GradCAM':
        gradcam = GradCAM(model_to_use, target_layers="features.17")
        return gradcam(in_tensor, class_idx)
    elif method == 'GradCAMpp':
        gradcampp = GradCAMpp(model_to_use, target_layers="features.17")
        return gradcampp(in_tensor, class_idx)
    elif method == 'CAM':
        cam = CAM(model_to_use, target_layers="features.17")
        return cam(in_tensor, class_idx)
    elif method == 'GuidedBackpropGrad':
        gbp = GuidedBackpropGrad(model_to_use)
        return gbp(in_tensor, class_idx)
    elif method == 'GuidedBackpropSmoothGrad':
        gbps = GuidedBackpropSmoothGrad(model_to_use)
        return gbps(in_tensor, class_idx)
    elif method == 'SmoothGrad':
        sg = SmoothGrad(model_to_use)
        return sg(in_tensor, class_idx)
    elif method == 'VanillaGrad':
        vg = VanillaGrad(model_to_use)
        return vg(in_tensor, class_idx)
    else:
        raise ValueError(f"Unknown MONAI method: {method}")

def compute_sbsm(query_tensor, neighbor_tensor, model, block_size=24, stride=12):
    """
    SBSM (Similarity-Based Saliency Map) for any CNN that outputs flat feature vectors.
    Includes debug logging. Works for VGG, DenseNet, etc.
    """
    device = next(model.parameters()).device
    query_tensor = query_tensor.to(device)
    neighbor_tensor = neighbor_tensor.to(device)

    # Extract base embeddings
    with torch.no_grad():
        # Handle ensemble models
        if hasattr(model, 'models') and isinstance(model.models, list):
            query_feat = torch.mean(torch.stack([m(query_tensor) for m in model.models]), dim=0)
            base_feat = torch.mean(torch.stack([m(neighbor_tensor) for m in model.models]), dim=0)
        else:
            query_feat = model(query_tensor)
            base_feat = model(neighbor_tensor)

        query_feat = query_feat.flatten(1)
        base_feat = base_feat.flatten(1)
        base_dist = F.pairwise_distance(query_feat, base_feat, p=2).item()

    # Prepare saliency map
    _, _, H, W = neighbor_tensor.shape
    saliency = torch.zeros(H, W).to(device)
    count = torch.zeros(H, W).to(device)

    # Generate occlusion masks
    mask_batch = []
    positions = []
    for y in range(0, H - block_size + 1, stride):
        for x in range(0, W - block_size + 1, stride):
            mask = torch.ones_like(neighbor_tensor)
            mask[:, :, y:y + block_size, x:x + block_size] = 0
            mask_batch.append(mask)
            positions.append((y, x))

    if not mask_batch:
        raise RuntimeError("No masks generated. Check block_size and stride vs. input image dimensions.")

    # Batch processing
    mask_batch = torch.cat(mask_batch, dim=0).to(device)
    repeated_neighbor = neighbor_tensor.repeat(mask_batch.shape[0], 1, 1, 1)
    masked_imgs = repeated_neighbor * mask_batch

    with torch.no_grad():
        # Handle ensemble models
        if hasattr(model, 'models') and isinstance(model.models, list):
            masked_feats = torch.mean(torch.stack([m(masked_imgs) for m in model.models]), dim=0)
        else:
            masked_feats = model(masked_imgs)
            
        masked_feats = masked_feats.flatten(1)
        dists = F.pairwise_distance(query_feat.expand_as(masked_feats), masked_feats, p=2)

    for idx, (y, x) in enumerate(positions):
        dist_drop = max(dists[idx].item() - base_dist, 0)
        importance_mask = torch.zeros(H, W).to(device)
        importance_mask[y:y + block_size, x:x + block_size] = 1
        saliency += dist_drop * importance_mask
        count += importance_mask

    # Normalize and smooth
    saliency = saliency / (count + 1e-8)
    saliency = saliency.cpu().numpy()
    saliency = np.maximum(saliency, 0)
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = gaussian_filter(saliency, sigma=min(block_size // 6, 3))

    return saliency

def compute_cam_pytorch(in_tensor, reference_embedding, nn_module, target_layer_name="features.17"):
    """Modified CAM computation for VGG16_Base_224 with ensemble support"""
    try:
        # Handle ensemble models - use first model for CAM
        if hasattr(nn_module, 'models') and isinstance(nn_module.models, list):
            model_to_use = nn_module.models[0]
        else:
            model_to_use = nn_module
        
        # Verify target layer
        try:
            if isinstance(target_layer_name, str):
                module = model_to_use.features
                for part in target_layer_name.split('.')[1:]:
                    module = getattr(module, part)
                target_layer = module
            else:
                target_layer = target_layer_name
        except Exception as e:
            raise RuntimeError(f"Layer access failed: {str(e)}")

        # Hook setup
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        with torch.set_grad_enabled(True):
            features = model_to_use.features(in_tensor)
            pooled_features = model_to_use.adaptive_pool(features)
            output = pooled_features.view(pooled_features.size(0), -1)

            # Simulate relevance via similarity
            sim_score = -torch.norm(output - reference_embedding.unsqueeze(0), p=2)
            model_to_use.zero_grad()
            sim_score.backward(retain_graph=True)

        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()

        # Verify activations/gradients
        if not activations or not gradients:
            raise RuntimeError("No activations or gradients captured")

        # Compute CAM
        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Enhanced normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(in_tensor.shape[2], in_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().detach().cpu()

    except Exception as e:
        raise RuntimeError(f"CAM computation failed: {str(e)}")

# Function: Compute InputXGradient
#def compute_input_x_gradient(model, in_tensor, q_target):
    # Enable gradients on input tensor
    #in_tensor.requires_grad_()
    # Build InputXGradient object
    #IxG = InputXGradient(model)
    # Get attribution
    #attribution = IxG.attribute(inputs=in_tensor, target=q_target)
    #return attribution

# Function: Compute GuidedGradCam
#def compute_guided_grad_cam(model, model_last_conv_layer, in_tensor, q_target):
    # Enable gradients on input tensor
    #in_tensor.requires_grad_()
    # Build GuidedGradCam object
    #GGC = GuidedGradCam(model, model_last_conv_layer)
    # Get attribution
    #attribution = GGC.attribute(inputs=in_tensor, target=q_target)
    #return attribution

def get_xai_attribution(model, in_tensor, method='IntegratedGradients', backend='Captum', reference_tensor=None, **kwargs):
    """Unified XAI computation for both backends with ensemble support"""
    if backend == 'Captum':
        if method == 'IntegratedGradients':
            if reference_tensor is None:
                reference_tensor = in_tensor
            return compute_integrated_gradients(
                model=model,
                query_tensor=reference_tensor,
                neighbor_tensor=in_tensor
            )
        else:
            return compute_attributions(
                model=model,
                in_tensor=in_tensor,
                target=0,  # For classification-style methods
                method=method,
                **kwargs
            )
    elif backend == 'MONAI':
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is not available. Please install monai package.")
        return compute_monai_results(
            in_tensor=in_tensor,
            class_idx=0,
            method=method,
            model=model
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

def process_cam_to_heatmap(cam, img_tensor):
    """Convert CAM output to heatmap"""
    cam_np = cam.numpy()
    if len(cam_np.shape) == 3:
        cam_np = np.mean(cam_np, axis=0)
    cam_np = np.squeeze(cam_np)
    cam_np = np.maximum(cam_np, 0)
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np

def process_ig_to_heatmap(attributions):
    """Convert IG attributions to heatmap"""
    attr_np = attributions.detach().cpu().squeeze().numpy()
    if attr_np.ndim == 3:
        attr_np = np.mean(attr_np, axis=0)
    attr_np = (attr_np - attr_np.mean()) / (attr_np.std() + 1e-8)
    attr_np = np.clip(attr_np, -3, 3)
    return (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

def process_sbsm_to_heatmap(saliency_map):
    """Process SBSM output to standardized heatmap"""
    saliency_map = np.maximum(saliency_map, 0)
    if saliency_map.max() > 0:
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    return saliency_map