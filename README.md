# LoRA Stable Diffusion for Streaming Applications

A comprehensive project for fine-tuning Stable Diffusion models using LoRA (Low-Rank Adaptation) techniques, optimized for streaming diffusion applications.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using LoRA
- **Streaming Optimization**: Optimized for real-time streaming applications
- **Multiple Model Support**: Support for various Stable Diffusion versions
- **Custom Dataset Training**: Tools for preparing and training on custom datasets
- **Model Optimization**: Techniques for faster inference and reduced memory usage
- **Streaming Integration**: Ready-to-use streaming diffusion pipeline

## Project Structure

```
├── configs/              # Training and model configurations
├── data/                # Dataset management and preprocessing
├── models/              # Model definitions and checkpoints
├── scripts/             # Training and utility scripts
├── src/                 # Core source code
├── streaming/           # Streaming-specific implementations
├── experiments/         # Experiment tracking and results
├── tests/              # Unit and integration tests
├── docker/             # Docker configurations
└── notebooks/          # Jupyter notebooks for exploration
```

## Quick Start

1. **Setup Environment**
   ```bash
   conda create -n lora-sd python=3.9
   conda activate lora-sd
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   ```bash
   # Extract frames from videos
   python scripts/extract_frames.py --video_path ./data/raw/video.mp4 --output_dir ./data/raw/frames --interval 30
   
   # Resize images to training resolution
   python scripts/resize_images.py --input_dir ./data/raw/frames --output_dir ./data/processed/train --resolution 512
   
   # Or prepare dataset from existing images
   python scripts/prepare_dataset.py --data_dir ./data/raw --output_dir ./data/processed
   ```

3. **Start Training**
   ```bash
   python scripts/train_lora.py --config configs/lora_base.yaml
   ```

4. **Test Streaming**
   ```bash
   python streaming/stream_server.py --model_path ./models/checkpoints/best_model.pth
   ```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended)
- 8GB+ VRAM for training

## License

MIT License
