from dataclasses import dataclass
from typing import List, Dict

@dataclass
class wandb_cfg:
    project: str
    entity: str

@dataclass
class split_ratio_cfg:
    train: float
    val: float
    test: float
    random_seed: int

@dataclass
class dataset_cfg:
    img_size: int
    in_channels: List[str]
    msk_file_ext: List[str]
    img_file_ext: List[str]
    split_ratio: split_ratio_cfg
    batch_size: int
    num_workers: int

@dataclass
class ipath_cfg:
    CFG_DIR: str
    DTA_DIR: str
    HIST_DIR: str
    IMG_DTA_DIR: str
    MSK_DTA_DIR: str

@dataclass
class opath_cfg:
    BIN_DIR: str
    ART_DIR: str
    DTA_BIN_DIR: str
    MDL_BIN_DIR: str
    RES_BIN_DIR: str
    PLT_DIR: str
    GRF_PLT_DIR: str
    SMP_PLT_DIR: str
    LOG_DIR: str
    SLR_LOG_DIR: str
    MDL_LOG_DIR: str

@dataclass
class path_cfg:
    PROJECT_DIR: str
    ipath: ipath_cfg
    opath: opath_cfg

@dataclass
class valid_channels_cfg:
    valid_channels: List[str]
    
@dataclass
class valid_models_cfg:
    valid_types: List[str]

@dataclass
class valid_loss_fn_ord_cfg:
    valid_fns: List[str]

@dataclass
class valid_loss_fn_nrm_cfg:
    valid_fns: List[str]

@dataclass
class typecheck_cfg:
    valid_channels: valid_channels_cfg
    valid_models: valid_models_cfg
    valid_loss_fn_ord: valid_loss_fn_ord_cfg
    valid_loss_fn_nrm: valid_loss_fn_nrm_cfg

@dataclass
class param_cfg:
    mdl_type: str
    img_size: int
    in_channels: List[str]
    lr: float
    weight_decay: float
    use_weights: bool
    loss_fn: str
    num_epochs: int

@dataclass
class setup_cfg:
    dataset: dataset_cfg
    dev_type: str
    models: Dict[str, param_cfg]
    typecheck: typecheck_cfg
    path: path_cfg
    auto_manage_artifacts: bool
    wandb: wandb_cfg

# Color mapping for debug visualization
COLOR_MAP = {
    2: [255, 0, 0],    # Right breast - red
    3: [0, 255, 0],    # Left breast - green
    4: [0, 0, 255],    # Right nipple - blue
    5: [255, 255, 0]   # Left nipple - yellow
}

COLOR_TO_CLASS = {
    (255, 0, 0): 2,    # Red -> Right breast
    (0, 255, 0): 3,    # Green -> Left breast
    (0, 0, 255): 4,    # Blue -> Right nipple
    (255, 255, 0): 5   # Yellow -> Left nipple
}