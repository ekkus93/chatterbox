import torch
import torch.nn as nn
from typing import Dict, Any
from .int8_layers import QuantizedLinear, QuantizedEmbedding


def quantize_model(model: nn.Module, skip_layers=None) -> nn.Module:
    """
    Quantize a PyTorch model to INT8
    
    Args:
        model: The model to quantize
        skip_layers: List of layer names to skip quantization
    
    Returns:
        Quantized model
    """
    if skip_layers is None:
        skip_layers = []
    
    def _quantize_module(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if full_name in skip_layers:
                continue
                
            if isinstance(child, nn.Linear):
                # Replace Linear with QuantizedLinear
                quantized_linear = QuantizedLinear.from_float(child)
                setattr(module, child_name, quantized_linear)
                print(f"Quantized Linear layer: {full_name}")
                
            elif isinstance(child, nn.Embedding):
                # Replace Embedding with QuantizedEmbedding
                quantized_embedding = QuantizedEmbedding.from_float(child)
                setattr(module, child_name, quantized_embedding)
                print(f"Quantized Embedding layer: {full_name}")
                
            else:
                # Recursively quantize child modules
                _quantize_module(child, full_name)
    
    # Create a copy of the model to avoid modifying the original
    quantized_model = type(model)(model.hp if hasattr(model, 'hp') else None)
    quantized_model.load_state_dict(model.state_dict())
    
    # Quantize the model
    _quantize_module(quantized_model)
    
    return quantized_model


def save_quantized_model(model: nn.Module, save_path: str, metadata: Dict[str, Any] = None):
    """Save a quantized model with metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'quantization_info': {
            'method': 'int8_per_channel',
            'quantized_layers': []
        }
    }
    
    # Add metadata
    if metadata:
        save_dict.update(metadata)
    
    # Record which layers are quantized
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
            save_dict['quantization_info']['quantized_layers'].append(name)
    
    torch.save(save_dict, save_path)
    print(f"Quantized model saved to: {save_path}")


def load_quantized_model(model_class, load_path: str, device='cpu'):
    """Load a quantized model"""
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    # Handle PyTorch dynamic quantization format (model object directly saved)
    if isinstance(checkpoint, torch.nn.Module):
        print(f"Loading PyTorch dynamic quantized model from: {load_path}")
        
        # Restore the device property that gets lost during serialization
        def device_property(self):
            try:
                return self.speech_head.weight().device
            except:
                # Fallback: find any parameter and return its device
                for param in self.parameters():
                    if torch.is_tensor(param):
                        return param.device
                return torch.device('cpu')
        
        # Override the device property on the class
        checkpoint.__class__.device = property(device_property)
        
        return checkpoint, {'method': 'pytorch_dynamic_quantization'}
    
    # Handle custom quantization format (original logic)
    else:
        # Initialize the model (assuming it has quantized layers)
        if 'hp' in checkpoint:
            model = model_class(checkpoint['hp'])
        else:
            model = model_class()
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded custom quantized model from: {load_path}")
        print(f"Quantized layers: {checkpoint['quantization_info']['quantized_layers']}")
        
        return model, checkpoint.get('quantization_info', {})
