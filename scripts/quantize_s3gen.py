#!/usr/bin/env python3
"""
Script to quantize S3Gen model to INT8
"""
import argparse
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from chatterbox.models.s3gen import S3Gen
from chatterbox.quantization import quantize_model, save_quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize S3Gen model to INT8')
    parser.add_argument('--input', required=True, help='Path to input S3Gen model (.pt file)')
    parser.add_argument('--output', required=True, help='Path to output quantized model')
    parser.add_argument('--skip-layers', nargs='*', default=[], 
                       help='Layer names to skip quantization')
    
    args = parser.parse_args()
    
    print(f"Loading S3Gen model from: {args.input}")
    
    # Load the original model
    checkpoint = torch.load(args.input, map_location='cpu', weights_only=False)
    
    # Initialize S3Gen model
    s3gen_model = S3Gen()
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model'][0] if isinstance(checkpoint['model'], list) else checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    s3gen_model.load_state_dict(state_dict, strict=False)
    s3gen_model.eval()
    
    print("Original model size:")
    total_params = sum(p.numel() for p in s3gen_model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # Quantize the model
    print("\nQuantizing model to INT8...")
    quantized_model = quantize_model(s3gen_model, skip_layers=args.skip_layers)
    
    # Calculate quantized model size
    int8_params = 0
    fp32_params = 0
    for name, param in quantized_model.named_parameters():
        if 'int8' in name:
            int8_params += param.numel()
        else:
            fp32_params += param.numel()
    
    for name, buffer in quantized_model.named_buffers():
        if 'int8' in name:
            int8_params += buffer.numel()
        elif 'scale' in name or 'bias' in name:
            fp32_params += buffer.numel()
    
    estimated_size = (int8_params * 1 + fp32_params * 4) / (1024**2)
    print(f"\nQuantized model size:")
    print(f"  INT8 parameters: {int8_params:,}")
    print(f"  FP32 parameters: {fp32_params:,}")
    print(f"  Estimated size: {estimated_size:.2f} MB")
    print(f"  Compression ratio: {(total_params * 4) / (int8_params * 1 + fp32_params * 4):.2f}x")
    
    # Save quantized model
    save_quantized_model(quantized_model, args.output)
    print(f"\nQuantized S3Gen model saved to: {args.output}")


if __name__ == "__main__":
    main()
