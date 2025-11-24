import time
import torch
import numpy as np
from thop import profile
from model import create_model


def measure_efficiency(model, device="cpu", input_size=(1, 3, 224, 224)):

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    for _ in range(10):
        _ = model(dummy_input)

    iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    
    end_time = time.time()
    total_time = end_time - start_time
    
    avg_latency = (total_time / iterations) * 1000
    throughput = iterations / total_time

    print(f"\nEfficiency Metrics ({model.__class__.__name__}):")
    print(f"  Device: {device}")
    print(f"  Input Size: {input_size}")
    print(f"  Total Params: {total_params:,}")
    print(f"  Trainable Params: {trainable_params:,}")
    print(f"  FLOPs: {flops / 1e9:.2f} G")
    print(f"  Latency: {avg_latency:.2f} ms")
    print(f"  Throughput: {throughput:.2f} img/sec")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'avg_latency': avg_latency,
        'throughput': throughput
    }

if __name__ == "__main__":
    # Test with both models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        
    print(f"Running efficiency tests on {device}...")
    
    # 1. ResNet18
    print("\n--- ResNet18 ---")
    resnet, _, _ = create_model(model_name='resnet18', device=device)
    measure_efficiency(resnet, device=device)
    
    # 2. DeiT-Tiny
    print("\n--- DeiT-Tiny ---")
    deit, _, _ = create_model(model_name='deit_tiny', device=device)
    measure_efficiency(deit, device=device)

    