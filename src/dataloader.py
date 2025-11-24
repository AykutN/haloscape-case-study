import json
import os
import random
from collections import Counter, defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_ANALYSIS_DIR = '../reports/data_analysis'


def _to_rgb(image):
    return image.convert('RGB')


def get_data_transforms(image_size: int = 224):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Lambda(_to_rgb),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(_to_rgb),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def _stratified_split_indices(labels: List[int], val_split: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    rng = random.Random(seed)
    train_indices, val_indices = [], []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_split)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = 224,
):
    train_transform, val_transform = get_data_transforms(image_size=image_size)

    base_dataset = datasets.ImageFolder(root=data_dir)
    train_indices, val_indices = _stratified_split_indices(base_dataset.targets, val_split, seed)

    train_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    return train_loader, val_loader, base_dataset.classes


def create_test_loader(data_dir: str, batch_size: int = 32, num_workers: int = 4, image_size: int = 224):
    _, val_transform = get_data_transforms(image_size=image_size)

    test_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    use_pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    print(f"Test Batches: {len(test_loader)}")

    return test_loader


def analyze_dataset(data_dir: str, save_dir: str = DEFAULT_ANALYSIS_DIR, samples_per_class: int = 4, max_mode_samples: int = 200):
    os.makedirs(save_dir, exist_ok=True)

    dataset = datasets.ImageFolder(root=data_dir)
    class_names = dataset.classes

    class_counts = Counter(dataset.targets)
    df = pd.DataFrame({
        'Class': [class_names[idx] for idx in sorted(class_counts.keys())],
        'Count': [class_counts[idx] for idx in sorted(class_counts.keys())],
    })

    df.to_csv(os.path.join(save_dir, 'class_distribution.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df['Class'], df['Count'], color='steelblue', edgecolor='black')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Image count')
    plt.title('Class distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    mode_counts = Counter()
    for path, _ in dataset.samples[:max_mode_samples]:
        with Image.open(path) as img:
            mode_counts[img.mode] += 1

    mode_df = pd.DataFrame({'Mode': list(mode_counts.keys()), 'Count': list(mode_counts.values())})
    mode_df.to_csv(os.path.join(save_dir, 'image_modes.csv'), index=False)

    class_to_paths = {idx: [] for idx in range(len(class_names))}
    for path, label in dataset.samples:
        if len(class_to_paths[label]) < samples_per_class:
            class_to_paths[label].append(path)
        if all(len(paths) == samples_per_class for paths in class_to_paths.values()):
            break

    n_rows = len(class_names)
    fig, axes = plt.subplots(n_rows, samples_per_class, figsize=(3 * samples_per_class, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row_idx, class_name in enumerate(class_names):
        for col_idx in range(samples_per_class):
            ax = axes[row_idx][col_idx]
            if col_idx < len(class_to_paths[row_idx]):
                img_path = class_to_paths[row_idx][col_idx]
                with Image.open(img_path) as img:
                    ax.imshow(img, cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10, fontweight='bold')

    plt.suptitle('Sample images per class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'sample_images.png'), dpi=300)
    plt.close()

    summary = {
        'total_images': len(dataset),
        'classes': class_names,
        'class_counts': df.set_index('Class')['Count'].to_dict(),
        'image_modes': dict(mode_counts),
        'analysis_dir': os.path.abspath(save_dir),
    }

    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Dataset analysis exported to {save_dir}")

    return summary


if __name__ == "__main__":
    data_dir = "../data/Training"
    analyze_dataset(data_dir)
    loaders = create_dataloaders(data_dir, batch_size=16, val_split=0.2)
    train_loader, val_loader, classes = loaders
    print(f"Classes: {classes}")
