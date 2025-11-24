import torch
import torch.nn as nn
from torchvision import models

def load_pretrained_resnet18(num_classes=4):
    """
    Load pretrained ResNet18 and replace final layer
    
    Args:
        num_classes: Number of output classes (4 for brain tumors)
    
    Returns:
        model: ResNet18 with modified final layer
    """ 
    
    # Load pretrained ResNet18 (trained on ImageNet)
    # weights='DEFAULT' loads the best available weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Check original fc layer
    in_features = model.fc.in_features  # ResNet18 fc: 512 input features
    
    # Replace final layer: 512 -> 4 (instead of 512 -> 1000)
    model.fc = nn.Linear(in_features, num_classes)
    
    print(f"Pretrained ResNet18 loaded")
    print(f"Final layer: {in_features} -> {num_classes}")
    
    return model
