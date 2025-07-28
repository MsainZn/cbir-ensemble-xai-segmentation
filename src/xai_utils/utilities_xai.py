# PyTorch Imports
import torch
import torch.nn as nn  
from PIL import Image
from monai.visualize import CAM, GradCAM, GradCAMpp, GuidedBackpropGrad, GuidedBackpropSmoothGrad, SmoothGrad, VanillaGrad

# Captum Imports
from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    GuidedGradCam,
)

# Function: Compute IntegratedGradients
def compute_integrated_gradients(model, query_tensor, neighbor_tensor, baseline=None, steps=50):
    """
    Compute IG attributions using L2 distance (consistent with retrieval).
    """
    if baseline is None:
        baseline = 0 * neighbor_tensor  # Black image baseline
    
    neighbor_tensor.requires_grad_()
    
    def similarity_fn(neighbor_input):
        with torch.no_grad():
            query_embed = model(query_tensor)
        neighbor_embed = model(neighbor_input)
        return -torch.norm(query_embed - neighbor_embed, p=2).unsqueeze(0)  # Negative for maximization
    
    ig = IntegratedGradients(similarity_fn)
    
    # Stable attribution computation
    attributions = ig.attribute(
        inputs=neighbor_tensor,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
        return_convergence_delta=False  # Avoid tensor issues
    )
    
    # Verification
    with torch.no_grad():
        original_dist = -similarity_fn(neighbor_tensor).item()  # Get actual L2 distance
        print(f"\n[DEBUG] Original L2 distance: {original_dist:.4f}")
        
        mask = (attributions.abs() > attributions.abs().mean()).float()
        masked_dist = -similarity_fn(neighbor_tensor * mask).item()
        print(f"[DEBUG] Masked L2 distance: {masked_dist:.4f}")
        print(f"[DEBUG] Distance increase: {masked_dist - original_dist:.4f}")
    
    return attributions

def get_xai_attribution(model, in_tensor, method, backend='Captum', reference_tensor=None):
    """Unified XAI computation for both backends"""
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
                method=method
            )
    else:
        return compute_monai_results(
            in_tensor=in_tensor,
            class_idx=0,
            method=method,
            model=model
        )

def compute_sbsm(query_tensor, neighbor_tensor, model, block_size=24, stride=12):
    """
    SBSM (Similarity-Based Saliency Map) for any CNN that outputs flat feature vectors.
    Includes debug logging. Works for VGG, DenseNet, etc. Not MONAI, need to remove it
    """
    device = next(model.parameters()).device
    query_tensor = query_tensor.to(device)
    neighbor_tensor = neighbor_tensor.to(device)

    print("[DEBUG] Running SBSM with block_size =", block_size, ", stride =", stride)

    # Extract base embeddings
    with torch.no_grad():
        query_feat = model(query_tensor)
        base_feat = model(neighbor_tensor)

        print("[DEBUG] Output shapes - query:", query_feat.shape, ", neighbor:", base_feat.shape)

        query_feat = query_feat.flatten(1)
        base_feat = base_feat.flatten(1)
        base_dist = F.pairwise_distance(query_feat, base_feat, p=2).item()
        print("[DEBUG] L2 distance between query and neighbor:", base_dist)

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

    print(f"[DEBUG] Total masked patches to evaluate: {len(mask_batch)}")

    # Batch processing
    mask_batch = torch.cat(mask_batch, dim=0).to(device)
    repeated_neighbor = neighbor_tensor.repeat(mask_batch.shape[0], 1, 1, 1)
    masked_imgs = repeated_neighbor * mask_batch

    with torch.no_grad():
        masked_feats = model(masked_imgs).flatten(1)
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

    print("[DEBUG] Final saliency map shape:", saliency.shape)
    return saliency




def compute_cam_pytorch(in_tensor, reference_embedding, nn_module, target_layer_name="features.17"):
    #Modified CAM computation for VGG16_Base_224
    try:
        # 1. Model preparation
        model_to_use = nn_module
        print(f"[DEBUG] Model type: {type(model_to_use).__name__}")
        
        # 2. Verify target layer
        print(f"[DEBUG] Target layer: {target_layer_name}")
        try:
            if isinstance(target_layer_name, str):
                module = model_to_use.features  # Access features directly
                for part in target_layer_name.split('.')[1:]:  # Skip 'features' prefix
                    module = getattr(module, part)
                target_layer = module
            else:
                target_layer = target_layer_name
            print(f"[DEBUG] Layer found: {target_layer}")
            print(f"[DEBUG] Layer type: {type(target_layer).__name__}")
            print(f"[DEBUG] Layer weight shape: {target_layer.weight.shape if hasattr(target_layer, 'weight') else 'N/A'}")
        except Exception as e:
            print(f"[DEBUG] Layer access failed: {str(e)}")
            raise

        # 3. Hook setup
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            print(f"[DEBUG] Forward hook - output shape: {output.shape}")
            activations.append(output.detach())
            
        def backward_hook(module, grad_input, grad_output):
            print(f"[DEBUG] Backward hook - grad_output[0] shape: {grad_output[0].shape}")
            gradients.append(grad_output[0].detach())
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        print("[DEBUG] Hooks registered")

        # 4. Forward pass - modified for VGG16_Base_224
        with torch.set_grad_enabled(True):
            print("[DEBUG] Running forward pass...")
            
            # Get features from the model
            features = model_to_use.features(in_tensor)
            pooled_features = model_to_use.adaptive_pool(features)
            output = pooled_features.view(pooled_features.size(0), -1)
            
            print(f"[DEBUG] Model output shape: {output.shape}")

            # Simulate relevance via similarity
            sim_score = -torch.norm(output - reference_embedding.unsqueeze(0), p=2) 

            model_to_use.zero_grad()
            print("[DEBUG] Running backward pass using L2 distance...")
            sim_score.backward(retain_graph=True)
            print("[DEBUG] Backward pass completed")

        # 5. Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        print("[DEBUG] Hooks removed")

        # 6. Verify activations/gradients
        if not activations:
            print("[ERROR] No activations captured!")
            raise RuntimeError("No activations captured")
        if not gradients:
            print("[ERROR] No gradients captured!")
            raise RuntimeError("No gradients captured")
            
        print(f"[DEBUG] Activations shape: {activations[0].shape}")
        print(f"[DEBUG] Gradients shape: {gradients[0].shape}")

        # 7. Compute CAM
        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        print(f"[DEBUG] Weights shape: {weights.shape}")
        
        cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
        cam = torch.relu(cam)
        print(f"[DEBUG] Raw CAM stats - min: {cam.min().item():.4f}, max: {cam.max().item():.4f}")

        # 8. Enhanced normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(in_tensor.shape[2], in_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        print(f"[DEBUG] Normalized CAM stats - min: {cam.min().item():.4f}, max: {cam.max().item():.4f}")
        
        return cam.squeeze().detach().cpu()

    except Exception as e:
        print(f"[ERROR] CAM computation failed: {str(e)}")
        raise


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


# Function: Compute Attributions
def compute_attributions(model, in_tensor, q_target, method, **kwargs):
    device = next(model.parameters()).device
    in_tensor = in_tensor.to(device)
    assert method in (
        'IntegratedGradients',
        'InputXGradient',
        'GuidedGradCam',
    ), 'Please provide a valid method.'

    # Select attribution method
    if method == 'IntegratedGradients':
        attribution = compute_integrated_gradients(model, in_tensor, q_target)
    elif method == 'InputXGradient':
        attribution = compute_input_x_gradient(model, in_tensor, q_target)
    elif method == 'GuidedGradCam':
        attribution = compute_guided_grad_cam(model, kwargs['model_last_conv_layer'], in_tensor, q_target)
    return attribution.to(device)

def get_xai_attribution(model, in_tensor, method, backend='Captum', reference_tensor=None):
    """Unified XAI computation for both backends"""
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
                method=method
            )
    else:
        return compute_monai_results(
            in_tensor=in_tensor,
            class_idx=0,
            method=method,
            model=model
        )

def process_cam_to_heatmap(cam, img_tensor):
    cam_np = cam.numpy()
    if len(cam_np.shape) == 3:
        cam_np = np.mean(cam_np, axis=0)
    cam_np = np.squeeze(cam_np)
    cam_np = np.maximum(cam_np, 0)
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np

def process_ig_to_heatmap(attributions):
    attr_np = attributions.detach().cpu().squeeze().numpy()
    if attr_np.ndim == 3:
        attr_np = np.mean(attr_np, axis=0)
    attr_np = (attr_np - attr_np.mean()) / (attr_np.std() + 1e-8)
    attr_np = np.clip(attr_np, -3, 3)
    return (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
