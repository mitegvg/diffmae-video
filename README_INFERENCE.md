# DiffMAE Inference on CPU

This README provides instructions for running inference with a pre-trained DiffMAE (Diffusion Masked Autoencoder) model on CPU.

## Model Overview

DiffMAE combines concepts from Masked Autoencoders (MAE) and Diffusion Models to reconstruct images where parts have been masked out. The model works as follows:

1. The input image is split into patches
2. A percentage of these patches are randomly masked
3. The encoder processes the visible patches
4. The diffusion model is used to reconstruct the masked regions

## Requirements

Make sure you have the following dependencies installed:
```
torch
torchvision
numpy
Pillow
matplotlib
```

## Running Inference

You can run inference using the provided `inference_cpu.py` script:

```bash
python inference_cpu.py --model_path /path/to/model_checkpoint.pth --image_path /path/to/input_image.jpg
```

### Command-line Arguments

- `--model_path`: Path to the pre-trained model checkpoint (required)
- `--image_path`: Path to the input image file (required)
- `--output_path`: Path to save the output image (default: output.png)
- `--mask_ratio`: Percentage of image patches to mask (default: 0.5)
- `--img_size`: Size to resize the input image (default: 224)
- `--seed`: Random seed for reproducibility (default: 42)

## Output

The script will generate two files:
1. `output.png`: A side-by-side comparison of the original image, mask visualization, and reconstructed image
2. `output_plot.png`: A matplotlib visualization with labeled subplots

## Example

```bash
python inference_cpu.py --model_path checkpoints/model_epoch_100.pth --image_path examples/cat.jpg --mask_ratio 0.6
```

This will:
1. Load the model from 'checkpoints/model_epoch_100.pth'
2. Load the image 'examples/cat.jpg'
3. Mask 60% of the image patches
4. Run the diffusion reconstruction process
5. Save visualizations of the results