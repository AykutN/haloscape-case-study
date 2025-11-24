import sys
import os

# Add the parent directory to sys.path to allow importing from 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.resnet18 import load_pretrained_resnet18
from models.deit_tiny import load_pretrained_deit

def apply_freeze_strategy(model, strategy='partial'):
    """
    Apply layer freezing strategy for both ResNet and DeiT
    """
    # Detect model type
    is_vit = hasattr(model, 'head')
    final_layer_name = 'head' if is_vit else 'fc'
    final_layer = getattr(model, final_layer_name)
    
    if strategy == 'all':
        # Freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only final layer
        for param in final_layer.parameters():
            param.requires_grad = True
            
        print(f"Strategy: Feature Extraction (only {final_layer_name} trainable)")
    
    elif strategy == 'partial':
        if is_vit:
            # DeiT Strategy: Freeze patch_embed and first 8 blocks
            # Train last 4 blocks (8-11) and head
            layers_to_freeze = ['patch_embed', 'pos_drop', 'cls_token', 'pos_embed']
            for i in range(8):
                layers_to_freeze.append(f'blocks.{i}.')
            
            for name, param in model.named_parameters():
                should_freeze = any(layer in name for layer in layers_to_freeze)
                param.requires_grad = not should_freeze
                
            print("Strategy: Partial Fine-tuning (ViT)")
            print("  Frozen: patch_embed, blocks 0-7")
            print("  Trainable: blocks 8-11, norm, head")
            
        else:
            # ResNet Strategy: Freeze early layers
            layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']
            
            for name, param in model.named_parameters():
                should_freeze = any(layer in name for layer in layers_to_freeze)
                param.requires_grad = not should_freeze
                
            print("Strategy: Partial Fine-tuning (ResNet)")
            print("  Frozen: conv1, bn1, layer1, layer2")
            print("  Trainable: layer3, layer4, fc")
    
    elif strategy == 'none':
        for param in model.parameters():
            param.requires_grad = True
        print("Strategy: Full Fine-tuning (all layers trainable)")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {frozen_params:,}")
    
    return model


def get_optimizer(model, base_lr=0.0001, strategy='partial'):
    """
    Create optimizer with differential learning rates
    """
    # Detect model type
    is_vit = hasattr(model, 'head')
    final_layer_name = 'head' if is_vit else 'fc'
    
    if strategy == 'partial' or strategy == 'none':
        # Differential learning rates
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if final_layer_name in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        # Use AdamW instead of Adam for better weight decay handling (crucial for ViT)
        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': base_lr * 0.1},  # Smaller LR for pretrained
            {'params': new_params, 'lr': base_lr}  # Base LR for new layer
        ], weight_decay=0.01)
        
        print(f"\nOptimizer: AdamW")
        print(f"  Pretrained layers LR: {base_lr * 0.1}")
        print(f"  New {final_layer_name} layer LR: {base_lr}")
        print(f"  Weight Decay: 0.01")
    
    else:  # strategy == 'all'
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=0.01
        )
        
        print(f"\nOptimizer: AdamW")
        print(f"  Learning rate: {base_lr}")
        print(f"  Weight Decay: 0.01")
    
    return optimizer


def get_scheduler(optimizer, step_size=7, gamma=0.1):
    """
    Create learning rate scheduler
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    
    print(f"\nScheduler: StepLR")
    print(f"  Reduce LR by {gamma}x every {step_size} epochs")
    
    return scheduler


def create_model(model_name, num_classes=4, freeze_strategy='partial', base_lr=0.0001, device='cpu'):
    """
    Main function: Create complete model with optimizer and scheduler
    """
    
    print("=" * 70)
    print(f"MODEL CREATION: {model_name}")
    print("=" * 70)
    
    # 1. Load pretrained model
    if model_name == 'resnet18':
        model = load_pretrained_resnet18(num_classes)
    elif model_name == 'deit_tiny':
        model = load_pretrained_deit(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 2. Apply freeze strategy
    model = apply_freeze_strategy(model, freeze_strategy)
    
    # 3. Move to device
    model = model.to(device)
    print(f"\nModel moved to: {device}")
    
    # 4. Create optimizer
    optimizer = get_optimizer(model, base_lr, freeze_strategy)
    
    # 5. Create scheduler
    scheduler = get_scheduler(optimizer)
    
    print("\n" + "=" * 70)
    print("MODEL READY FOR TRAINING")
    print("=" * 70)
    
    return model, optimizer, scheduler


if __name__ == "__main__":
    # Test the model creation
    print("Testing model creation...\n")
    
    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Create ResNet
    print("\n--- Testing ResNet18 ---")
    create_model(model_name='resnet18', device=device)
    
    # Create DeiT
    print("\n--- Testing DeiT-Tiny ---")
    create_model(model_name='deit_tiny', device=device)

