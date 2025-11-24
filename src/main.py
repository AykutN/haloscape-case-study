import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import json
from PIL import Image

# Import custom modules
from model import create_model
from train import train_model
from efficiency import measure_efficiency

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_smart_split(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Splits dataset based on visual similarity clustering to prevent data leakage.
    Images that are very similar (likely same patient) are kept in the same split.
    """
    print("Analyzing dataset for potential leakage (Smart Split)...")
    
    # 1. Extract simplified features (thumbnails) for fast comparison
    indices = list(range(len(dataset)))
    thumbnails = []
    labels = []
    
    # Create a temporary loader for fast reading
    # We use a small size for clustering to be fast
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Generating perceptual hashes...")
    for img, label in tqdm(loader):
        # Downsample to 8x8 grayscale for "perceptual hashing" like comparison
        thumb = torch.nn.functional.interpolate(img, size=(8, 8), mode='bilinear')
        thumb = thumb.mean(dim=1).view(-1).numpy() # Flatten
        thumbnails.append(thumb)
        labels.append(label.item())
        
    thumbnails = np.array(thumbnails)
    
    # 2. Simple Clustering: Group images with MSE < threshold
    # Since O(N^2) is too slow for 3000 images (9M comparisons), we use a greedy approach
    # Sort by L1 norm of the thumbnail to reduce search space? 
    # Or just use a simpler heuristic: Sequential filenames often imply same patient.
    # But we don't have filenames easily here without hacking ImageFolder.
    # Let's use Agglomerative Clustering from sklearn if available, or a greedy bucket sort.
    
    from sklearn.cluster import AgglomerativeClustering
    
    # Distance threshold: 0.05 (determined empirically from check_leakage)
    # If MSE < 0.05, they are likely same patient.
    # AgglomerativeClustering uses euclidean distance (L2). MSE = L2^2 / N.
    # So L2 threshold = sqrt(MSE * N). N=64. sqrt(0.05 * 64) = sqrt(3.2) ~= 1.8
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=2.0, # Slightly loose to be safe
        metric='euclidean', 
        linkage='average'
    )
    
    print("Clustering similar images...")
    cluster_labels = clustering.fit_predict(thumbnails)
    n_clusters = len(set(cluster_labels))
    print(f"Found {n_clusters} unique visual clusters (potential patients) from {len(dataset)} images.")
    
    # 3. Split Clusters instead of Images
    # Group indices by cluster
    cluster_to_indices = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
        cluster_to_indices[cluster_id].append(idx)
        
    # Shuffle clusters
    unique_clusters = list(cluster_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(unique_clusters)
    
    # Assign clusters to sets
    train_cutoff = int(n_clusters * train_ratio)
    val_cutoff = int(n_clusters * (train_ratio + val_ratio))
    
    train_clusters = unique_clusters[:train_cutoff]
    val_clusters = unique_clusters[train_cutoff:val_cutoff]
    test_clusters = unique_clusters[val_cutoff:]
    
    train_indices = [idx for c in train_clusters for idx in cluster_to_indices[c]]
    val_indices = [idx for c in val_clusters for idx in cluster_to_indices[c]]
    test_indices = [idx for c in test_clusters for idx in cluster_to_indices[c]]
    
    print(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def save_history(history, save_path):
    # Convert numpy types to python types for JSON serialization
    serializable_history = {}
    for k, v in history.items():
        serializable_history[k] = [float(x) for x in v]
        
    with open(save_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)

def plot_training_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def evaluate_model(model, test_loader, device, class_names, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'test', 'confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score']
    }

def run_experiment(model_name, config, device, train_loader, val_loader, test_loader, class_names):
    print(f"\nRunning Experiment for: {model_name}")
    
    # Create Model
    model, optimizer, scheduler = create_model(
        model_name, 
        num_classes=len(class_names),
        device=device
    )
    
    # Setup paths
    save_path = os.path.join(config['save_dir'], f"{model_name}_best.pth")
    results_dir = os.path.join('reports', model_name)
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, 'training_history.json')
    
    # Train
    print(f"Starting training for {config['num_epochs']} epochs...")
    start_time = time.time()
    
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs=config['num_epochs'],
        save_path=save_path,
        class_weights=config['class_weights']
    )
    
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds")
    
    # Save History
    save_history(history, history_path)
    
    # Plot Curves
    plot_training_curves(history, results_dir)

    # Measure Efficiency (Post-training)
    print(f"\nMeasuring Efficiency for {model_name}...")
    eff_metrics = measure_efficiency(model, device)

    # Evaluate on Test Set
    print(f"\nEvaluating on Test Set...")
    test_metrics = evaluate_model(model, test_loader, device, class_names, results_dir)

    return {
        'model': model_name,
        'val_acc': max(history['val_acc']),
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['macro_f1'],
        'training_time_sec': training_time,
        'best_model_path': save_path,
        'params': eff_metrics['total_params'],
        'flops_g': eff_metrics['flops'] / 1e9,
        'latency_ms': eff_metrics['avg_latency'],
        'throughput': eff_metrics['throughput']
    }

def main():
    # Setup
    config = {
        'data_dir': 'data',
        'save_dir': 'src', # Save models in src for now
        'batch_size': 32,
        'num_epochs': 30, # Kept at 30, augmentations should help
        'seed': 42
    }
    
    set_seed(config['seed'])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms (Augmented for Train)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Subtle jitter for MRI
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    print("Loading Data...")
    # We load the full dataset with TRAIN transforms first, but this applies transforms to val/test too if we split subsets directly.
    # To do it correctly with Subsets, we need a custom Dataset class or just accept that val/test might have transforms (bad) 
    # OR we rely on the fact that transforms are applied at __getitem__.
    # The standard way with Subsets is to wrap them.
    
    full_dataset = datasets.ImageFolder(os.path.join(config['data_dir'], 'Training'), transform=train_transform)
    
    # Create Smart Split
    train_subset, val_subset, test_subset = create_smart_split(full_dataset)
    
    # IMPORTANT: Overwrite transforms for Val and Test subsets
    # Subsets don't have 'transform' attribute, they delegate to dataset.
    # We need to create copies of the underlying dataset with different transforms.
    # But since we have indices, we can just create new Subsets pointing to a Clean Dataset.
    
    clean_dataset = datasets.ImageFolder(os.path.join(config['data_dir'], 'Training'), transform=val_test_transform)
    val_subset = Subset(clean_dataset, val_subset.indices)
    test_subset = Subset(clean_dataset, test_subset.indices)
    
    # Loaders
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")

    # Calculate Class Weights (based on Train subset only)
    print("Calculating class weights...")
    train_targets = [full_dataset.targets[i] for i in train_subset.indices]
    class_counts = np.bincount(train_targets)
    total_samples = len(train_targets)
    class_weights = torch.FloatTensor(total_samples / (len(class_names) * class_counts)).to(device)
    config['class_weights'] = class_weights
    print(f"Class Weights: {class_weights}")

    # Run Experiments
    results = []
    
    # ResNet-18
    results.append(run_experiment('resnet18', config, device, train_loader, val_loader, test_loader, class_names))
    
    # DeiT-Tiny
    results.append(run_experiment('deit_tiny', config, device, train_loader, val_loader, test_loader, class_names))
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv('reports/comparison_results.csv', index=False)
    print("\nExperiments Completed. Results saved to reports/comparison_results.csv")
    print(df)

if __name__ == "__main__":
    main()
