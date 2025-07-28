import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from transformers import AutoImageProcessor, SegformerImageProcessor
from transformers import MobileNetV2ForSemanticSegmentation, SegformerForSemanticSegmentation, \
    MobileViTV2ForSemanticSegmentation, BeitForSemanticSegmentation, UperNetForSemanticSegmentation, \
    DPTForSemanticSegmentation

from transformers import SamModel, SamProcessor

# Grab Models 1) https://huggingface.co/docs/transformers/v4.27.0/en/tasks/semantic_segmentation
# Grab Models 2) https://huggingface.co/models?pipeline_tag=image-segmentation&library=pytorch,transformers&sort=downloads
# Construct UNET style architecture with native **PyTorch**

# POTENTIAL MODELS FOR LATER SegGptModel, CLIPSegForImageSegmentation, Data2VecVisionForSemanticSegmentation

class BaseUNet(nn.Module, ABC):
    def __init__(self,
                 in_channel_count:int,
                 num_class:int, 
                 img_size:int,
                 name:str="BaseUNet", 
                 is_ordinal:bool=False,
                 mdl_lbl:str = None):
        
        super().__init__()
        self.is_ordinal = is_ordinal
        self.in_channel_count = in_channel_count
        self.num_output = num_class - 1 if is_ordinal else num_class
        self.mdl_name = name    # The name Hugging face gives
        self.mdl_lbl = mdl_lbl  # The label based on user Config
        self.num_class = num_class
        self.img_size = img_size

    def concat_channels(self, x:dict[str, torch.Tensor])-> torch.Tensor:
        return torch.cat([x[k] for k in x], dim=1)

    def calc_model_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad
                   and p.dim() > 1)  # Only count trainable parameters

    @abstractmethod
    def forward(self, x):
        pass  # Each subclass must implement its own forward method

    def predict_logits(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        self.train() if is_train else self.eval()
        # self.to(x.device)
        with torch.inference_mode():
            logits = self.forward(x)
        return logits
    
    def predict_with_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_ordinal:
            probs  = torch.sigmoid(logits)
            output = torch.sum(probs > 0.5, dim=1)
        else:
            probs  = F.softmax(logits, dim=1)
            output = torch.argmax(probs, dim=1) 
        return output

    def predict(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        logits = self.predict_logits(x, is_train)
        if self.is_ordinal:
            probs = torch.sigmoid(logits)
            return torch.sum(probs > 0.5, dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1)

    def predict_one(self, x:dict[str, torch.Tensor], is_train:bool=False) -> torch.Tensor:
        return self.predict(x, is_train).squeeze(0)

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size=3, padding=1, Dropout:float=0.3):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Dropout),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(Dropout),
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channel_count, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channel_count, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channel_count, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel_count, in_channel_count//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel_count, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class CustomUNet(BaseUNet):
    def __init__(self, in_channel_count:int,num_class:int, img_size:int, 
                 name:str="CustomUNet", is_ordinal:bool=False, mdl_lbl:str = None):
        
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.down_conv_1 = DownSample(self.in_channel_count, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=self.num_output, kernel_size=1)

    def forward (self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_conv_1(b, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        logits = self.out(up_4)
        return logits
 
class LightUNet(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, 
                 img_size:int, name:str="LightUNet", 
                 is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
            
        self.down_conv_1 = DownSample(self.in_channel_count, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)

        self.bottle_neck = DoubleConv(256, 512)

        self.up_conv_1 = UpSample(512, 256)
        self.up_conv_2 = UpSample(256, 128)
        self.up_conv_3 = UpSample(128, 64)

        self.out = torch.nn.Conv2d(in_channels=64, 
                                   out_channels=self.num_output, 
                                   kernel_size=1)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)

        b = self.bottle_neck(p3)

        up_1 = self.up_conv_1(b,    down_3)
        up_2 = self.up_conv_2(up_1, down_2)
        up_3 = self.up_conv_3(up_2, down_1)

        logits = self.out(up_3)

        return logits

# ..................Hugging Face Models Started..................

class MobileNetV2_DeepLab(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="google/deeplabv3_mobilenet_v2_1.0_513", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        # Load model with the specified number of output classes
        self.model = MobileNetV2ForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        
        # Image processor (optional, if needed for pre/post-processing)
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits  # (batch, num_class, H, W)

class Segformer_Face(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="jonathandinu/face-parsing", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)
        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)

class Segformer_Nvidia(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", is_ordinal:bool=False, mdl_lbl:str = None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)
        
        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)

class Segformer_MITb0(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="nvidia/segformer-b0-finetuned-ade-512-512", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = SegformerImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)
    
class MobileViTV2_Apple(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="apple/mobilevitv2-1.0-imagenet1k-256", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = MobileViTV2ForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)
    
class DPT_INTEL(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="Intel/dpt-large-ade", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = DPTForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)
    
class UperNet_Openmmlab(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int,
                 name:str="openmmlab/upernet-convnext-tiny", is_ordinal:bool=False, mdl_lbl:str=None):
        # in_channel_count is only 3
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            self.mdl_name, num_labels=self.num_output, ignore_mismatched_sizes=True # THIS PART IS NOT DEFINED, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = x['rgb']
        self.to(x.device)
        logits = self.model(x).logits  # Raw output (batch, num_class, H, W)

        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)

class BEIT_Microsoft(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=640, 
                 name:str="microsoft/beit-base-finetuned-ade-640-640", 
                 is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(in_channel_count, num_class, img_size, name, is_ordinal, mdl_lbl)
        
        self.model = BeitForSemanticSegmentation.from_pretrained(
            self.mdl_name, image_size=self.img_size, num_labels=self.num_output, 
            ignore_mismatched_sizes=True, num_channels=self.in_channel_count
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.mdl_name)

    def forward(self, x):
        x = self.concat_channels(x)
        self.to(x.device)
        outputs = self.model(x)  # Raw output (batch, num_class, H, W)
        logits = outputs.logits
        # Resize logits to match input spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # (H, W) from input
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        return logits  # (batch, num_class, H, W)
    
class SAM_MetaC(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Intermediate convolution
            torch.nn.BatchNorm2d(512),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),   # Further convolution
            torch.nn.BatchNorm2d(256),  # Added BatchNorm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )
        self.final_conv = torch.nn.Conv2d(256, self.num_output, kernel_size=1) 

    def forward(self, x):
        x = x['rgb']
        self.to(x.device)
        features = self.encoder(x).last_hidden_state  
        segmentation = self.decoder(features)  
        segmentation = self.final_conv(segmentation)  
        segmentation = F.interpolate(segmentation, size=(1024, 1024), mode="bilinear", align_corners=False)
        return segmentation

class SAM_MetaT(BaseUNet):
    def __init__(self, in_channel_count:int, num_class:int, img_size:int=1024, 
                 name:str="facebook/sam-vit-base", is_ordinal:bool=False, mdl_lbl:str=None):
        super().__init__(3, num_class, img_size, name, is_ordinal, mdl_lbl)

        self.model = SamModel.from_pretrained(self.mdl_name)
        self.image_processor = SamProcessor.from_pretrained(self.mdl_name)
        self.encoder = self.model.vision_encoder
        
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(d_model=256, nhead=8),
            num_layers=6
        )
        self.final_conv = torch.nn.Conv2d(256, self.num_output, kernel_size=1)

        # Normalization layers
        self.decoder_norm = torch.nn.LayerNorm(256)  # Normalize the decoder output

    def forward(self, x):
        x = x['rgb']
        self.to(x.device)
        # Encoder output is B, 256, 64, 64
        features = self.encoder(x).last_hidden_state  
        # Reshape for Transformer decoder: (B, C, H*W) → (H*W, B, C)
        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        segmentation = self.decoder(features, features) 
        segmentation = self.decoder_norm(segmentation)  # Normalize the decoder output
        segmentation = segmentation.permute(1, 2, 0).view(B, C, H, W)

        segmentation = self.final_conv(segmentation) 
        segmentation = torch.functional.F.interpolate(segmentation, size=(1024, 1024), mode="bilinear", align_corners=False)
        return segmentation

class Model_Factory:
    # Create a dictionary that maps model names to model classes
    MODEL_FACTORY: dict[str: BaseUNet] = {
        'LightUNet': LightUNet,
        'CustomUNet': CustomUNet,
        'BEIT_Microsoft': BEIT_Microsoft,
        'Segformer_Face': Segformer_Face,
        'DPT_INTEL': DPT_INTEL,
        'Segformer_MITb0': Segformer_MITb0,
        'Segformer_Nvidia': Segformer_Nvidia,
        'MobileNetV2_DeepLab': MobileNetV2_DeepLab,
        'MobileViTV2_Apple': MobileViTV2_Apple,
        'UperNet_Openmmlab': UperNet_Openmmlab,
        'SAM_MetaC': SAM_MetaC,
        'SAM_MetaT': SAM_MetaT,
        # 'LightUNetResidualDepth': LightUNetResidualDepth,
        # 'LightUNetAttentionDepth': LightUNetAttentionDepth
    }

    @classmethod
    def create_model(cls, input_mdl:str, **kwargs)->BaseUNet:
        """
        Create a model based on the provided configuration.
        """
        # This method should be implemented to create and return a model instance
        if input_mdl not in cls.MODEL_FACTORY:
            raise ValueError(f"Model {input_mdl} not found in the factory.")
        return cls.MODEL_FACTORY[input_mdl](**kwargs)

    @classmethod
    def get_model_list(cls)->list[str]:
        """
        Get the list of available models in the factory.
        """
        return list(cls.MODEL_FACTORY.keys())


# Model loading section
def load_sam_model(model_path: str, device: str = None) -> SAM_MetaC:
    """Load SAM_MetaC model with proper initialization and verification"""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SAM_MetaC(
        in_channel_count=3,
        num_class=6,
        img_size=1024,
        is_ordinal=False
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    # Verify model
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
        output = model({'rgb': dummy_input})
        assert output.shape == (1, 6, 1024, 1024), "Model output shape mismatch"
    
    return model

#Other model loading functions can be added similarly