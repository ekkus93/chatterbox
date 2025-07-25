import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store quantized weights and scales
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features, dtype=torch.float32))
        self.register_buffer('input_scale', torch.tensor(1.0))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer('bias', None)
    
    @classmethod
    def from_float(cls, float_linear):
        """Convert a regular Linear layer to INT8 quantized version"""
        quantized = cls(float_linear.in_features, float_linear.out_features, 
                       bias=float_linear.bias is not None)
        
        # Quantize weights per-channel (output channel)
        weight = float_linear.weight.data
        weight_scales = weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        weight_scales = torch.clamp(weight_scales, min=1e-8)
        quantized_weight = torch.round(weight / weight_scales).clamp(-128, 127).to(torch.int8)
        
        quantized.weight_int8.copy_(quantized_weight)
        quantized.weight_scale.copy_(weight_scales.squeeze())
        
        if float_linear.bias is not None:
            quantized.bias.copy_(float_linear.bias.data)
        
        return quantized
    
    def forward(self, x):
        # Dequantize weights on-the-fly
        weight_fp = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
        return F.linear(x, weight_fp, self.bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.register_buffer('weight_int8', torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(num_embeddings, dtype=torch.float32))
    
    @classmethod
    def from_float(cls, float_embedding):
        """Convert a regular Embedding layer to INT8 quantized version"""
        quantized = cls(float_embedding.num_embeddings, float_embedding.embedding_dim)
        
        # Quantize embeddings per-token
        weight = float_embedding.weight.data
        weight_scales = weight.abs().max(dim=1, keepdim=True)[0] / 127.0
        weight_scales = torch.clamp(weight_scales, min=1e-8)
        quantized_weight = torch.round(weight / weight_scales).clamp(-128, 127).to(torch.int8)
        
        quantized.weight_int8.copy_(quantized_weight)
        quantized.weight_scale.copy_(weight_scales.squeeze())
        
        return quantized
    
    def forward(self, input):
        # Dequantize embeddings for selected indices
        selected_weights = self.weight_int8[input]  # (batch, seq, dim)
        selected_scales = self.weight_scale[input]  # (batch, seq)
        return selected_weights.float() * selected_scales.unsqueeze(-1)
