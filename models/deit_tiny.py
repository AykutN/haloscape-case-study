import torch
import torch.nn as nn
import timm

def load_pretrained_deit(num_classes=4):
    """
    Load pretrained DeiT-Tiny and replace final layer
    
    Args:
        num_classes: Number of output classes
    
    Returns:
        model: DeiT-Tiny with modified final layer
    """
    # Load pretrained DeiT-Tiny
    # deit_tiny_patch16_224 is the standard ImageNet-1k pretrained model
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    
    # Check original head layer
    # In timm/DeiT, the classification head is usually named 'head'
    in_features = model.head.in_features
    
    # Replace final layer
    model.head = nn.Linear(in_features, num_classes)
    
    print(f"Pretrained DeiT-Tiny loaded")
    print(f"Final layer: {in_features} -> {num_classes}")
    
    return model
