# cbir-feature-extractors
This repository contains a framework for **Content-Based Image Retrieval (CBIR)** with support for **explainability (XAI), and segmentation**.  
It provides training and testing pipelines for deep learning models, post-hoc explainability methods (e.g., saliency maps), and segmentation utilities.  

## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:Pedrofs9/cbir-ensemble-xai-segmentation.git
```

Then go to the repository's main directory:
```bash
$ cd cbir-ensemble-xai-segmentation
```
## Usage
### Train Model
To train the models:
```bash
$ python src/main_image.py {command line arguments}
```
or:
```bash
$ sbatch scripts/image/GC_ViT_224.sh
```
This script accepts the following command line arguments:
```bash
 --config_json: path of the config json containing the model details
 --pickles_path: path of the pickles containing the data
 --results_path: path to save the results
 --train_or_test: 'train'
 --gpu_id
```
### Test Model
To test the models:
```bash
$ python src/main_image.py {command line arguments}
```
or:
```bash
$ sbatch sbatch scripts/image/GC_ViT_224.sh
```
This script accepts the following command line arguments:
```bash
 --pickles_path: path of the pickles containing the data
 --checkpoin_path: path to the pretrained model
 --train_or_test 'test'
 --gpu_id
```
### XAI Saliency Maps
To generate saliency maps:
```bash
$ python src/main_image.py {command line arguments}
```
or:
```bash
$ sbatch scripts/explainability/GC_ViT_224.sh
```
This script accepts the following command line arguments:
```bash
 --visualizations_path: path to save the visualizations
 --pickles_path: path of the pickles containing the data
 --train_or_test: 'test' 
 --visualize_queries: query rankings visualization
 --visualize_triplets: triplets visualization
 --generate_xai: xai application (if not passed, triplets and rankings are shown without saliency maps)
 --max_visualizations: maximum number of visualizations to show
 --results_path
 --checkpoint_path: path to the pretrained model
 --xai_backend: {Captum, MONAI} 
 --xai_method: {IntegratedGradients, CAM, SBSM} (CAM and SBSM only available with MONAI backend 
 --xai_batch_size: batch size during XAI application
 --gpu_id
```

### Segmentation
For segmentation application:
```bash
$ python src/segmentation_utils/SegmentationApplication.py {command line arguments}
```
or:
```bash
$ sbatch scripts/segmentation/segmentation.sh
```
This script accepts the following command line arguments:
```bash
  --input_folder: path to folder with images to segment
  --output_segmented: path to folder to save segmented images
  --output_debug: path to save segmentation debug
  --segmentation_model_path: path to the pretrained segmentation model
```

For segmentation overlap:
```bash
$ python src/segmentation_utils/SegmentationOverlap.py {command line arguments}
```
or:
```bash
$ sbatch scripts/segmentation/mask_overlap.sh
```
This script accepts the following command line arguments:
```bash
  --new_original_folder: path to the original images
  --mask_folder: path to the masks
  --output_folder: path to save the overlap results
```

For segmentation postprocessing:
```bash
$ python src/segmentation_utils/SegmentationPostProcessing.py {command line arguments}
```
or:
```bash
$ sbatch scripts/segmentation/postproc.sh
```
This script accepts the following command line arguments:
```bash
  --input_folder: path to the folder with the segmented masks
  --output_folder: path to save the results of mask post processing
```

For creating bounding box images containing segmented breasts:
```bash
$ python src/segmentation_utils/SegmentationBoundingBox.py {command line arguments}
```
or:
```bash
$ sbatch scripts/segmentation/bounding_box.sh
```
This script accepts the following command line arguments:
```bash
  --input_folder: path to the folder with the segmented masks
  --output_folder: path to save the resulting bounding boxes
```


## Credits and Acknowledgements
https://github.com/MsainZn/bcs-aesth-mmodal-retrieval

https://github.com/TiagoFilipeSousaGoncalves/bcs-aesth-mm-bcos-mir

