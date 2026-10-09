#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a YOLO11 oriented bounding-box periodontitis detector from a YAML configuration."""

from ultralytics import YOLO
import argparse
import yaml
import os
import torch

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='YOLO11 OBB Periodontitis detection model training')
    parser.add_argument('--config', type=str, default='training/cfgs/yolov11_obb.yaml',
                        help='Training configuration path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')
    args = parser.parse_args()
    
    # Load the configuration.
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file does not exist: {args.config}")
    
    config = load_config(args.config)
    
    # Check GPU availability.
    device = config.get('device', 0)
    if device != 'cpu':
        if torch.cuda.is_available():
            print(f"Using GPU device: cuda:{device}")
        else:
            print("Warning: no GPU detected; using CPU")
            config['device'] = 'cpu'
    else:
        print("Using CPU device")
    
    # Check required files.
    data_yaml = config.get('data', 'yolov11_obb_data.yaml')
    model_path = config.get('model', 'ultralytics/ultralytics/cfg/models/11/yolo11-obb.yaml')
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file does not exist: {data_yaml}")
    
    # Ultralytics downloads missing .pt pretrained models automatically.
    if not model_path.endswith('.pt') and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model configuration file does not exist: {model_path}")
    
    pretrained_weights = config.get('pretrained_weights')

    # Load the OBB model.
    print(f"\n{'='*60}")
    print(f"Load model: {model_path}")
    print(f"Task type: {config.get('task', 'obb')}")
    
    # Check for a pretrained .pt checkpoint.
    if model_path.endswith('.pt'):
        print("Detected pretrained checkpoint(.pt)")
        print("Loading the pretrained OBB model directly...")
        # Ultralytics automatically identifies the task when loading an OBB checkpoint.
        model = YOLO(model_path)
    else:
        # Specify the task explicitly when constructing a model from YAML.
        print("Creating the OBB model from its YAML configuration...")
        model = YOLO(model_path, task='obb')
        if pretrained_weights:
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights do not exist: {pretrained_weights}")
            print(f"Load pretrained weights: {pretrained_weights}")
            model.load(pretrained_weights)
    

    
    # Prepare training arguments without model, task or mode.
    # The task was set at model construction and need not be passed to train().
    train_args = {k: v for k, v in config.items() if k not in ['model', 'task', 'mode', 'pretrained_weights']}
    
    # Handle the resume argument.
    if args.resume:
        train_args['resume'] = True
        print("Resume training")
    
    # Print the training configuration.
    print(f"{'='*60}")
    print("Training configuration:")
    print(f"{'='*60}")
    print(f"Dataset configuration: {train_args['data']}")
    print(f"Training epochs: {train_args.get('epochs', 100)}")
    print(f"Batch size: {train_args.get('batch', 16)}")
    print(f"Image size: {train_args.get('imgsz', 640)}")
    print(f"Device: {train_args.get('device', 0)}")
    print(f"Optimizer: {train_args.get('optimizer', 'auto')}")
    print(f"Initial learning rate: {train_args.get('lr0', 0.01)}")
    print(f"Weight decay: {train_args.get('weight_decay', 0.0005)}")
    print(f"Worker threads: {train_args.get('workers', 8)}")
    project = train_args.get('project', 'runs/obb')
    name = train_args.get('name', 'train')
    print(f"Save path: {project}/{name}")
    print(f"{'='*60}\n")
    
    # Start training.
    print("Starting OBB training...")
    results = model.train(**train_args)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model output path: {results.save_dir}")
    print("="*60)
    
    # Validation.
    print("\nEvaluate the model on the validation split...")
    metrics = model.val()
    
    print("\n" + "="*60)
    print("Validation metrics:")
    print("="*60)
    if hasattr(metrics, 'box'):
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
    else:
        print(f"Metrics: {metrics}")
    print("="*60)
    
    # Save final results.
    results_file = os.path.join(results.save_dir, 'final_metrics.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Final Validation Metrics (OBB)\n")
        f.write(f"{'='*50}\n")
        if hasattr(metrics, 'box'):
            f.write(f"mAP50: {metrics.box.map50:.4f}\n")
            f.write(f"mAP50-95: {metrics.box.map:.4f}\n")
            f.write(f"Precision: {metrics.box.mp:.4f}\n")
            f.write(f"Recall: {metrics.box.mr:.4f}\n")
        else:
            f.write(f"Metrics: {metrics}\n")
    
    print(f"\nFinal metrics saved to: {results_file}")
    
    return results

if __name__ == '__main__':
    main()
