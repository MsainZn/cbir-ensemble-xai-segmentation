# Imports
from PIL import Image
import timm
import os
import subprocess
# PyTorch Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import BatchNorm1d
# Transformers Imports
from transformers import (
    AutoModel,
    AutoImageProcessor,
    ViTImageProcessor, 
    ViTModel, 
    DeiTImageProcessor, 
    DeiTModel, 
    BeitImageProcessor, 
    BeitModel, 
    Dinov2Model,
    AutoModelForImageClassification
)



# Class: Google_Base_Patch16_224
class Google_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained deit_tiny_patch16_224 ViT model
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  
        return featureVec



# Class: DeiT_Base_Patch16_224
class DeiT_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's embeddings



# Class: Beit_Base_Patch16_224
class Beit_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]



# Class: DinoV2_Base_Patch16_224
class DinoV2_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]



# Class: ResNet50_Base_224
class ResNet50_Base_224(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained ResNet50 model
        if weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1  # Use the default ImageNet weights
        # Load the pre-trained ResNet50 model
        base_model = resnet50(weights=weights)
        # Remove the final fully connected layer to use the model as a fixed feature extractor
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer (fc layer)
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 2048, 1, 1) from the average pooling layer
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 2048) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to convert the output to shape (batch_size, 2048)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        return x
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform



# Class: VGG16_Base_224
class VGG16_Base_224(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained VGG16 model
        if weights is not None:
            weights = VGG16_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained VGG16 model
        base_model = vgg16(weights=weights)
        
        # Remove the classifier layer to use the model as a fixed feature extractor
        # Here we keep all layers up to, but not including, the classifier layer.
        self.features = base_model.features  # Keep the convolutional feature extractor part
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 512, 7, 7)
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 512) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to resize the output to shape (batch_size, 512)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        return x
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform



# Class: VGG16_Base_224_MLP
class VGG16_Base_224_MLP(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT, feature_dim=512, embedding_size=256):
        super().__init__()
        # Load the pre-trained VGG16 model
        if weights is not None:
            weights = VGG16_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained VGG16 model
        base_model = vgg16(weights=weights)
        
        # Remove the classifier layer to use the model as a fixed feature extractor
        # Here we keep all layers up to, but not including, the classifier layer.
        self.features = base_model.features  # Keep the convolutional feature extractor part
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 512, 7, 7)
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 512) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Additional MLP Layers
        self.fc1 = nn.Linear(feature_dim * 49, embedding_size)  # 512*7*7 = 25088 inputs to 256
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size, 128)  # Second MLP layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)  # Third MLP layer

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to resize the output to shape (batch_size, 512, 7, 7)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Pass through the fully connected layers with ReLU activation
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        return x
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform



# Class: Google_Base_Patch16_224_MLP
class Google_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(Google_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained deit_tiny_patch16_224 ViT model
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer (change 768 to your feature size)
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Use outputs.last_hidden_state if no pooling
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: ResNet50_Base_224_MLP
class ResNet50_Base_224_MLP(nn.Module):
    def __init__(self, feature_dim=2048, embedding_size=512, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained ResNet50 model
        if weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained ResNet50 model
        base_model = resnet50(weights=weights)
                
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Adaptive pooling to make sure output size is consistent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer to transform the feature space
        self.fc1 = nn.Linear(feature_dim, embedding_size)
        
        # Add another layer, if needed, you can increase the complexity here
        self.fc2 = nn.Linear(embedding_size, 256)

        # Optional: Add a batch normalization layer
        self.batch_norm = nn.BatchNorm1d(256)

        # Optional: Add a Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features from the base model
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Pass through the first fully connected layer
        x = F.relu(self.fc1(x))

        # Pass through the second fully connected layer (with optional batch normalization and dropout)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = F.dropout(x, p=0.5, training=self.training)

        return x

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform



# Class: DeiT_Base_Patch16_224_MLP
class DeiT_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(DeiT_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained DEIT model
        self.feature_extractor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-patch16-224')
    
        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # Adjust the input size to match the output size of the last hidden layer of DeiT
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Further MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's embeddings
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: DinoV2_Base_Patch16_224_MLP
class DinoV2_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(DinoV2_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained DinoV2 model
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')

        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer; adjust the size to match DinoV2 output
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings from the [CLS] token, which is typically used for classification tasks
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: Beit_Base_Patch16_224_MLP
class Beit_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(Beit_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained Beit model
        self.feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer; size must match the output feature dimension of Beit
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings from the [CLS] token or equivalent
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: CrossViT_Tiny240
class CrossViT_Tiny240(nn.Module):

    # Method: __init__
    def __init__(self):
        super(CrossViT_Tiny240, self).__init__()
        model = timm.create_model(
            'crossvit_tiny_240.in1k', 
            pretrained=True, 
            num_classes=0
        )
        self.model = model
        return
    
    def get_transform(self):
        def transform(image_path):
            # self.model.eval()
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: LeViTConv256
class LeViTConv256(nn.Module):

    # Method: __init__
    def __init__(self):
        super(LeViTConv256, self).__init__()
        model = timm.create_model(
            'levit_conv_256.fb_dist_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: ConViT_Tiny
class ConViT_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(ConViT_Tiny, self).__init__()
        
        model = timm.create_model(
            'convit_tiny.fb_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model

        return


    # Method: get_transform
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform


    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: MaxViT_Tiny_224
class MaxViT_Tiny_224(nn.Module):

    # Method: __init__
    def __init__(self):
        super(MaxViT_Tiny_224, self).__init__()
        model = timm.create_model(
            'maxvit_tiny_tf_224.in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class MViTv2_Tiny
class MViTv2_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(MViTv2_Tiny, self).__init__()
        model = timm.create_model(
            'mvitv2_tiny.fb_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class DaViT_Tiny
class DaViT_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(DaViT_Tiny, self).__init__()
        model = timm.create_model(
            'davit_tiny.msft_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec

# Class DaViT_Base
class DaViT_Base(nn.Module):

    # Method: __init__
    def __init__(self):
        super(DaViT_Base, self).__init__()
        model = timm.create_model(
            "hf_hub:timm/davit_base.msft_in1k",
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec

# Class: Swin_Base_Patch4_Window7_224
class Swin_Base_Patch4_224(nn.Module):
    def __init__(self):
        super(Swin_Base_Patch4_224, self).__init__()
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=0  # Removes the classification head
        )
    
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform

    def forward(self, input):
        featureVec = self.model(input)
        return featureVec

# Class: ResNeXt50_32x4d
class ResNeXt50_32x4d_Base_224(nn.Module):
    def __init__(self):
        super(ResNeXt50_32x4d_Base_224, self).__init__()
        self.model = timm.create_model(
            'resnext50_32x4d',
            pretrained=True,
            num_classes=0  # Removes classification head for feature extraction
        )
        
        # Set target layer for CAM
        self.cam_target_layer = self._determine_target_layer()
        #print(f" ResNeXt Initialized with CAM target layer: {self.cam_target_layer}")
        self._verify_target_layer()

    def _determine_target_layer(self):
        """Find the best convolutional layer for CAM"""
        possible_layers = [
            "layer4.2.conv3",  # Last conv layer in ResNeXt50
            "model.layer4.2.conv3",
            self._find_last_conv_layer()  # Fallback
        ]
        
        for layer in possible_layers:
            try:
                module = self.model
                for part in layer.replace('model.', '').split('.'):
                    module = getattr(module, part)
                if isinstance(module, nn.Conv2d):
                    return layer
            except AttributeError:
                continue
                
        raise ValueError("Could not determine valid CAM target layer")

    def _find_last_conv_layer(self):
        """Find the last convolutional layer as fallback"""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers[-1] if conv_layers else None

    def _verify_target_layer(self):
        """Verify the target layer exists in the model"""
        module = self.model
        parts = self.cam_target_layer.replace('model.', '').split('.')
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                conv_layers = [name for name, m in self.model.named_modules() 
                             if isinstance(m, nn.Conv2d)]
                raise ValueError(
                    f"Target layer '{self.cam_target_layer}' not found.\n"
                    f"Available conv layers:\n{conv_layers[-5:]}"
                )
        
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"Target layer {self.cam_target_layer} is not a Conv2d layer")

    def get_target_layer(self):
        """Get the target layer module by name"""
        module = self.model
        for part in self.cam_target_layer.replace('model.', '').split('.'):
            module = getattr(module, part)
        return module

    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform

    def forward(self, input):
        featureVec = self.model(input)
        return featureVec

# Class: SwinV2_Base_Patch4_Window16_256
class SwinV2_Base_Patch4_Window16_256(nn.Module):

    def __init__(self):
        super(SwinV2_Base_Patch4_Window16_256, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-base-patch4-window16-256"
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/swinv2-base-patch4-window16-256"
        )
        # Remove classification head for feature extraction
        self.model.classifier = nn.Identity()
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Use processor instead of timm transforms
            image_trans = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
            return image_trans
        return transform

    def forward(self, input):
        # The SwinV2 model from transformers expects a different forward format
        outputs = self.model(input, output_hidden_states=True)
        # Use the last hidden state as features
        featureVec = outputs.hidden_states[-1][:, 0, :]  # Using cls token
        return featureVec

# Class: DenseNet121_Base_224
class DenseNet121_Base_224(nn.Module):
    def __init__(self):
        super(DenseNet121_Base_224, self).__init__()
        # Create model with consistent naming
        self.model = timm.create_model('densenet121', pretrained=True, num_classes=0)
        
        # Set target layer with flexible naming
        self.cam_target_layer = self._determine_target_layer()
        #print(f"DenseNet Initialized with CAM target layer: {self.cam_target_layer}")
        
        # Verify the layer exists
        self._verify_target_layer()

    def _determine_target_layer(self):
        """Flexibly determine the target layer name"""
        possible_layers = [
            "features.denseblock4.denselayer16.conv2",  # Original
            "model.features.denseblock4.denselayer16.conv2",  # Possible alternative
            self._find_last_conv_layer()  # Fallback, not sure if it works well
        ]
        
        for layer in possible_layers:
            try:
                module = self.model
                for part in layer.replace('model.', '').split('.'):
                    module = getattr(module, part)
                if isinstance(module, nn.Conv2d):
                    return layer
            except AttributeError:
                continue
                
        raise ValueError("Could not determine valid CAM target layer")

    def _find_last_conv_layer(self):
        """Find the last convolutional layer as fallback"""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers[-1] if conv_layers else None

    def _verify_target_layer(self):
        """Verify the target layer exists in the model"""
        module = self.model
        parts = self.cam_target_layer.replace('model.', '').split('.')
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                conv_layers = [name for name, m in self.model.named_modules() 
                             if isinstance(m, nn.Conv2d)]
                raise ValueError(
                    f"Target layer '{self.cam_target_layer}' not found.\n"
                    f"Available conv layers:\n{conv_layers[-5:]}"
                )
        
        if not isinstance(module, nn.Conv2d):
            raise ValueError(f"Target layer {self.cam_target_layer} is not a Conv2d layer")

    def get_target_layer(self):
        """Get the target layer module by name"""
        module = self.model
        for part in self.cam_target_layer.replace('model.', '').split('.'):
            module = getattr(module, part)
        return module

    def get_transform(self):
        """Image transformation pipeline"""
        def transform(image_path):
            # Use self.model here instead of self.base_model
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(
                **data_config,
                is_training=False
            )
            image = Image.open(image_path).convert('RGB')
            return transforms(image)
        return transform

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

# Class: GCViT_Base
class GC_ViT_224(nn.Module):
    def __init__(self):
        super(GC_ViT_224, self).__init__()
        self.model = timm.create_model(
            'gcvit_base',
            pretrained=True,
            num_classes=0  # Removes classification head
        )
        return
    
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.bn2 = nn.BatchNorm1d(2*hidden_dim) 
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim) 
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure input is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # First layer
        x = self.fc1(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.bn1(x)
        x = torch.sigmoid(x)
        
        # Second layer
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.sigmoid(x)
        
        # Third layer
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = torch.sigmoid(x)
        
        # Output layer
        x = self.fc4(x)
        return torch.sigmoid(x)

    # Method: get_transform
    def get_transform(self):
        def transform(tabular_vector):
            x = torch.tensor(tabular_vector, dtype=torch.float32).squeeze(0)
            return x
        return transform,mkjvgytfrdz12\

class ModelEnsemble(nn.Module):
    def __init__(self, model_names, checkpoint_paths=None, trainable=False, 
                models_dict=None, fusion_type='projection',
                mlp_dims=None, proj_dim=None, use_l2_norm=True,
                transformer_dim=512, nhead=4, num_layers=1,
                load_function=torch.load):  # Add load_function parameter
        super().__init__()
        self.models = nn.ModuleList()
        self.trainable = trainable
        self.fusion_type = fusion_type
        self.use_l2_norm = use_l2_norm
        self.model_names = model_names
        
        # Load individual models 
        for i, name in enumerate(model_names):
            model = models_dict[name]() if isinstance(models_dict[name], type) else models_dict[name]
            
            if checkpoint_paths and i < len(checkpoint_paths) and checkpoint_paths[i]:
                # Use the provided load function
                state_dict = load_function(checkpoint_paths[i])
                model.load_state_dict(state_dict, strict=False)
        
            if not trainable:
                for param in model.parameters():
                    param.requires_grad = False
            
            self.models.append(model)
    

        # Calculate total feature dimension 
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(next(self.models[0].parameters()).device)
            self.total_dim = sum(m(dummy_input).shape[1] for m in self.models)

        # Fusion method initialization
        if self.fusion_type == 'mlp':
            if mlp_dims is None:
                mlp_dims = [self.total_dim // 2, self.total_dim // 4]
            layers = []
            dims = [self.total_dim] + mlp_dims  # 3 elements (input + hidden+output)
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:  # Still skip activation on last layer
                    layers.append(nn.ReLU()) 
            
            self.fusion = nn.Sequential(*layers) 
        elif self.fusion_type == 'transformer':
            self.transformer_dim = transformer_dim
            self.feature_extractor = nn.Linear(self.total_dim, transformer_dim)
            decoder_layer = TransformerDecoderLayer(
                d_model=transformer_dim,
                nhead=nhead
            )
            self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(transformer_dim, proj_dim or self.total_dim // 2)
        else:  # Default projection
            self.proj_dim = proj_dim if proj_dim else self.total_dim // 2
            self.fusion = nn.Sequential(
                nn.Linear(self.total_dim, self.proj_dim),
                nn.ReLU()
            )
        self._transform = self.models[0].get_transform()

    def get_transform(self):
        return self._transform

    def forward(self, x):
        # Extract features from all models 
        features = [model(x) for model in self.models]
        if self.use_l2_norm:
            features = [F.normalize(f, p=2, dim=1) for f in features]
        concatenated = torch.cat(features, dim=1)
        
        # Fusion method dispatching
        if self.fusion_type == 'transformer':
            # Modified to match ModelWithTransformerHead
            query = self.feature_extractor(concatenated).unsqueeze(0)  # [1, B, D]
            memory = torch.zeros_like(query)  # Zero memory like in example
            output = self.transformer_decoder(query, memory).squeeze(0)  # [B, D]
            output = self.fc_out(output)
        else:
            output = self.fusion(concatenated)
            
        return F.normalize(output, p=2, dim=1) if self.use_l2_norm else output

# Dictionary: Models dictionary
MODELS_DICT = {

    # Image
    "Google_Base_Patch16_224":Google_Base_Patch16_224(),
    "DeiT_Base_Patch16_224":DeiT_Base_Patch16_224(),
    "Beit_Base_Patch16_224":Beit_Base_Patch16_224(),
 #   "DinoV2_Base_Patch16_224":DinoV2_Base_Patch16_224(),
 #   "ResNet50_Base_224":ResNet50_Base_224(),
    "VGG16_Base_224":VGG16_Base_224(),
 #   "CrossViT_Tiny240":CrossViT_Tiny240(),
 #   "LeViTConv256":LeViTConv256(),
    "ConViT_Tiny":ConViT_Tiny(),
 #   "MaxViT_Tiny_224":MaxViT_Tiny_224(),
 #   "MViTv2_Tiny":MViTv2_Tiny(),
    "DaViT_Tiny":DaViT_Tiny(),
    "Swin_Base_Patch4_224":Swin_Base_Patch4_224(),
    "ResNeXt50_32x4d_Base_224":ResNeXt50_32x4d_Base_224(),
    "DaViT_Base":DaViT_Base(),
    "SwinV2_Base_Patch4_Window16_256":SwinV2_Base_Patch4_Window16_256(),
    "GC_ViT_224":GC_ViT_224(),
    "DenseNet121_Base_224":DenseNet121_Base_224(),

    #"TabularMLP_773_200_20": TabularMLP(5, 200, 20)
    # Multimodal
 #   "Google_Base_Patch16_224_MLP":Google_Base_Patch16_224_MLP(),
 #   "DinoV2_Base_Patch16_224_MLP":DinoV2_Base_Patch16_224_MLP(),
 #   "Beit_Base_Patch16_224_MLP":Beit_Base_Patch16_224_MLP(),
 #   "DeiT_Base_Patch16_224_MLP":DeiT_Base_Patch16_224_MLP(),
 #   "ResNet50_Base_224_MLP":ResNet50_Base_224_MLP(),
 #   "VGG16_Base_224_MLP":VGG16_Base_224_MLP()

}
