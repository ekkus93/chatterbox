#!/usr/bin/env python3
"""
Example script showing how to quantize Chatterbox models
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from chatterbox.tts import ChatterboxTTS

def main():
    # First, download or load your models normally
    device = "cpu"  # or "cuda"
    
    print("Loading original models...")
    model = ChatterboxTTS.from_pretrained(device)
    
    print("\nTo quantize your models, run these commands:")
    print("1. Quantize T3 model:")
    print("   python scripts/quantize_t3.py --input path/to/t3_cfg.pt --output t3_cfg_int8.pt")
    
    print("2. Quantize S3Gen model:")
    print("   python scripts/quantize_s3gen.py --input path/to/s3gen.pt --output s3gen_int8.pt")
    
    print("\n3. Then load quantized models:")
    print("   model = ChatterboxTTS.from_local_quantized('path/to/quantized/models', device)")
    
    # You can also quantize specific layers only
    print("\n4. To skip certain layers from quantization:")
    print("   python scripts/quantize_t3.py --input t3_cfg.pt --output t3_cfg_int8.pt --skip-layers speech_head text_head")

if __name__ == "__main__":
    main()
