# StratoLAMP: Droplet Detection and Classification Pipeline

This repository contains the code used for droplet instance segmentation, classification, and counting in the StratoLAMP workflow.

## Associated Publication

This code accompanies the following publication:

**StratoLAMP: Label-free, multiplex digital loop-mediated isothermal amplification based on visual stratification of precipitate**  
M. Jin, J. Ding, Y. Zhou, J. Chen, Y. Wang, and Z. Li  
*Proceedings of the National Academy of Sciences* (PNAS), 2024, **121**(2): e2314030121  
DOI: [10.1073/pnas.2314030121](https://doi.org/10.1073/pnas.2314030121)

## Overview

The project builds on the Matterport Mask R-CNN framework and includes:

- Model configuration and dataset loader for droplet classes.
- Training script for custom datasets.
- Inference script for image folder processing.
- Counting utilities for class-wise droplet quantification.
- Sample images and example outputs.

Droplet classes used in this repository:

- `negative`
- `low_positive`
- `medium_positive`
- `high_positive`

## Repository Structure

- `train.py`: training entrypoint.
- `droplets.py`: dataset parsing and model configuration.
- `droplet_video_detect.py`: inference on image folders and export of annotated images plus JSON outputs.
- `count_multi_types.py`: class-wise counting from JSON results.
- `mrcnn/`: local Mask R-CNN implementation.
- `model/`: pretrained weights (`0503_mask_rcnn_droplet_0120.h5`).
- `Sample images/`: sample input images.
- `results/`: sample prediction outputs.

## Environment Setup

1. Create and activate a Python environment (recommended: Python 3.8).
2. Install dependencies required by Matterport Mask R-CNN (TensorFlow/Keras compatible with this codebase, OpenCV, scikit-image, imgaug, NumPy, etc.).
3. Ensure GPU settings and CUDA dependencies are configured if you plan to use GPU inference/training.

## Data Format

Training and validation data are expected in:

- `<dataset_root>/train/*.json`
- `<dataset_root>/val/*.json`

The loader in `droplets.py` reads LabelMe-style JSON containing:

- `imageData` (base64 image)
- `imageWidth`, `imageHeight`
- `shapes` with polygon points and labels

Expected labels for this implementation are the four classes listed above.

## Training

Run training with:

```bash
python train.py train --dataset <dataset_root> --weights <weights.h5_or_coco_or_last_or_imagenet> --logs <log_dir>
```

Notes:

- Update `N_TRAIN` and `N_VAL` in `droplets.py` to match your dataset size.
- `train.py` contains a two-stage schedule (`heads` then `all` layers).
- The script currently forces CPU via `CUDA_VISIBLE_DEVICES = "-1"`; edit this line in `train.py` if GPU training is desired.

## Inference

Run:

```bash
python droplet_video_detect.py
```

Before running, edit `droplet_video_detect.py` to set:

- `weights_path`
- `image_path`
- `save_path`

Outputs:

- Annotated images in `<save_path>/image/`
- LabelMe-style JSON in `<save_path>/json/`

## Counting

`count_multi_types.py` provides counting helpers over JSON outputs.

- Use `count_single_frame(json_path)` for one JSON file.
- Use the script main block for folder-level aggregation after updating `json_path`.

## Sample Inputs and Outputs

- Example inputs: `Sample images/`
- Example outputs: `results/`

These files can be used to verify that the pipeline runs end-to-end.

## Citation

If you use this code in research, please cite:

```text
Jin M, Ding J, Zhou Y, Chen J, Wang Y, Li Z. StratoLAMP: Label-free, multiplex digital loop-mediated isothermal amplification based on visual stratification of precipitate. Proc Natl Acad Sci U S A. 2024;121(2):e2314030121. doi:10.1073/pnas.2314030121
```

## Acknowledgment

This repository adapts the Mask R-CNN implementation from Matterport:

- https://github.com/matterport/Mask_RCNN
