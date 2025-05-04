# DiffMAE Video Processing on CPU

This README provides instructions for running video inference with a pre-trained DiffMAE (Diffusion Masked Autoencoder) model on CPU.

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
opencv-python
gdown
```

You can install them with:
```bash
pip install torch torchvision numpy Pillow opencv-python gdown
```

## Obtaining Pretrained Models

### Option 1: Automatic Download
The script can automatically download the pretrained models from Google Drive using the `--download_model` flag.

### Option 2: Manual Download
You can manually download the pretrained models from the original repository's Google Drive links:
- [Butterfly Model](https://drive.google.com/drive/folders/1aQV-QoQxpR7UQPKSv0kHsFnPQDTwR5Wm)
- [Hair Model](https://drive.google.com/drive/folders/1d2d7xJmKaiHw22RQnlMPLwuCXQa3liaa)

## Running Video Inference

The script can process a video file, applying the DiffMAE model to each frame:

```bash
python inference_cpu.py --video_path /path/to/video.mp4 --model_path /path/to/model_checkpoint.pth
```

Or with automatic model download:

```bash
python inference_cpu.py --video_path /path/to/video.mp4 --download_model
```

### Command-line Arguments

- `--model_path`: Path to the pre-trained model checkpoint (required if --download_model is not used)
- `--download_model`: Download a pretrained model automatically
- `--video_path`: Path to the input video file (required)
- `--output_dir`: Directory to save output frames and videos (default: output_frames)
- `--mask_ratio`: Percentage of image patches to mask (default: 0.5)
- `--frame_size`: Size to resize the video frames (default: 224)
- `--seed`: Random seed for reproducibility (default: 42)
- `--frame_interval`: Process every nth frame (default: 1)
- `--max_frames`: Maximum number of frames to process, 0 means all (default: 0)

## Output

The script will generate:
1. Three directories with frame-by-frame images:
   - `output_dir/original/`: Original video frames
   - `output_dir/masked/`: Frames with masked patches visualized
   - `output_dir/reconstructed/`: Frames with reconstructed patches
   
2. Three video files:
   - `output_dir/original.mp4`: Original video
   - `output_dir/masked.mp4`: Video with masked patches visualized
   - `output_dir/reconstructed.mp4`: Video with reconstructed patches

## Example

```bash
python inference_cpu.py --video_path examples/video.mp4 --model_path models/diffmae_butterfly.pth --mask_ratio 0.6 --frame_interval 2 --max_frames 100
```

This will:
1. Load the model from 'models/diffmae_butterfly.pth'
2. Process the video 'examples/video.mp4'
3. Mask 60% of the patches in each frame
4. Process every 2nd frame
5. Process a maximum of 100 frames
6. Save results to the 'output_frames' directory