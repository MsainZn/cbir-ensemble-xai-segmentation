import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    def __init__(self, weights: torch.Tensor = None, 
                 device: str = None):
        super().__init__()
        self.weights = weights
        self.device = torch.device(device) if device else None

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass 

class PairwiseRankingLoss(BaseLoss): # Does nothing with weights
    def __init__(self, weights: torch.Tensor = None, margin=1.0, device: str = None):
        super().__init__(weights=None, device=device)
        self.margin = margin  # Margin for ranking loss
    
    def forward(self, pred, target):
        """
        pred: (batch_size, num_classes-1, H, W) - Raw logits
        target: (batch_size, H, W) - Ordinal labels
        """
        batch_size, num_thresholds, H, W = pred.shape

        # Expand target to create threshold labels
        target_expanded = target.unsqueeze(1).expand(-1, num_thresholds, -1, -1)  # (B, K-1, H, W)
        thresholds = torch.arange(num_thresholds, device=target.device).view(1, -1, 1, 1)
        threshold_labels = (target_expanded > thresholds).float()  # Binary threshold indicators

        # Compute pairwise ranking loss (Hinge loss)
        loss = 0.0
        for i in range(num_thresholds - 1):
            diff = pred[:, i, :, :] - pred[:, i + 1, :, :]  # Difference between consecutive thresholds
            hinge_loss = relu(self.margin - diff)  # Apply hinge loss
            mask = (threshold_labels[:, i, :, :] != threshold_labels[:, i + 1, :, :]).float()  # Consider only differing pairs
            loss += torch.mean(hinge_loss * mask)  # Apply mask to valid pairs
            #loss += torch.mean(relu(self.margin - diff))  # Hinge loss to enforce ranking
        
        return loss / (num_thresholds - 1)  # Normalize

class OrdinalRegressionLoss(BaseLoss):
    def __init__(self, weights:torch.Tensor=None, device:str=None):
        super().__init__(weights=weights, device=device)

        # If weights is provided, make sure it's a tensor and on the correct device
        if weights is not None:
            self.weights.to(device)  
            # Combine class weights to create threshold weights
            threshold_weights = []
            for i in range(len(self.weights) - 1):
                threshold_weights.append(self.weights[i] + self.weights[i + 1])
            self.weights = torch.tensor(threshold_weights, dtype=torch.float32).to(device)
            self.weights = self.weights.view(-1, 1, 1)  # Reshape for broadcasting
            
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.weights)

    def forward(self, pred, target):
        """
        pred: (batch_size, num_classes-1, H, W) - Sigmoid activations
        target: (batch_size, H, W) - Ordinal labels
        """
        batch_size, num_thresholds, H, W = pred.shape

        # Expand label to match prediction shape and create threshold labels
        target_expanded = target.unsqueeze(1).expand(-1, num_thresholds, -1, -1) # (B, K-1, H, W)
        thresholds = torch.arange(num_thresholds, device=target.device).view(1, -1, 1, 1)
        threshold_labels = (target_expanded > thresholds).float().view(batch_size, num_thresholds, H, W)
        loss = self.bce(pred, threshold_labels)
        return loss
    
class CombinedOrdinalLoss(BaseLoss):
    def __init__(self, weights=None, margin=1.0, alpha=0.5, device=None):
        super().__init__(weights=weights, device=device)
        self.wbce_loss = OrdinalRegressionLoss(weights, device)
        self.ranking_loss = PairwiseRankingLoss(None, margin)
        self.alpha = alpha  # Weighting factor

    def forward(self, pred, target):
        loss_bce = self.wbce_loss(pred, target)
        loss_rank = self.ranking_loss(pred, target)
        return loss_bce + self.alpha * loss_rank  # Combine both losses
    
class IoULoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, smooth: float = 1e-6, device: str = None):
        
        super().__init__(weights=weights, device=device)
        self.smooth = smooth
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = pred.shape

        pred = torch.softmax(pred, dim=1)

        target = target.long()
        target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        pred = pred.view(B, C, -1)
        target = target.view(B, C, -1)

        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)

        if self.weights is not None:
            iou_loss = (1 - iou) * self.weights.view(1, C)  # Expand weights to match (B, C)
            return iou_loss.sum() / (self.weights.sum() + self.smooth)  # Avoid division by zero
        
        return (1 - iou).mean()

class DiceLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, smooth: float = 1e-6, device: str = None):
        super().__init__(weights=weights, device=device)
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape

        # Apply softmax for multi-class probabilities
        pred = torch.softmax(pred, dim=1)

        # Ensure target is long for one-hot encoding
        target = target.long()
        target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        # Flatten to (B, C, H*W)
        pred = pred.view(B, C, -1)
        target = target.view(B, C, -1)

        # Compute Dice Score
        intersection = (pred * target).sum(dim=2)
        dice = (2 * intersection + self.smooth) / (pred.sum(dim=2) + target.sum(dim=2) + self.smooth)

        # Apply class weights if provided
        if self.weights is not None:
            dice_loss = (1 - dice) * self.weights.view(1, C)
            return dice_loss.sum() / (self.weights.sum() + self.smooth)  # Normalize by valid weights
        
        return (1 - dice).mean()  # Average Dice loss over classes
    
class FocalLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, alpha:float=1, gamma:float=2, device: str = None):
        super().__init__(weights=weights, device=device)
        self.alpha = alpha  # Class balancing (e.g., [1.0, 2.0, 5.0, 10.0])
        self.gamma = gamma  # Focusing parameter
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, self.weights, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
class CrossEntrophyLoss(BaseLoss):
    def __init__(self, weights: torch.Tensor = None, device: str = None):
        super().__init__(weights=weights, device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, targets):
        return self.ce_loss(inputs, targets)

class Loss_Factory:
    # Create a dictionary that maps Loss names to Loss classes
    LOSS_FUNCS = {
        'ce': CrossEntrophyLoss,
        'ord-reg': OrdinalRegressionLoss,
        'pair-rank': PairwiseRankingLoss,
        'comb-ord': CombinedOrdinalLoss,
        'iou': IoULoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
    }

    @classmethod
    def create_loss(cls, loss_type: str, **kwargs) -> BaseLoss:
        if loss_type not in cls.LOSS_FUNCS:
            raise ValueError(f"Loss function '{loss_type}' is not recognized.")
        return cls.LOSS_FUNCS[loss_type](**kwargs)
