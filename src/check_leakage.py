import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def check_leakage():
    # 1. Setup Data
    data_dir = 'data'
    
    # Use simple transform for comparison (resize only, no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    print("Loading datasets...")
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Training'), transform=transform)
    
    # Reproduce split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # 2. Flatten images for quick nearest neighbor search (L2 distance)
    num_test_samples = 50
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    print(f"\nAnalyzing {num_test_samples} random test images against full training set...")
    
    # Pre-load training images into memory
    train_images = []
    print("Indexing training set...")
    for img, _ in tqdm(train_loader):
        train_images.append(img.squeeze().numpy())
    
    train_images = np.array(train_images) # (N, 224, 224)
    
    high_similarity_pairs = []
    
    count = 0
    for test_img, _ in test_loader:
        if count >= num_test_samples:
            break
            
        test_img_np = test_img.squeeze().numpy()
        
        # Calculate MSE with all training images
        mse_scores = np.mean((train_images - test_img_np)**2, axis=(1,2))
        
        # Find closest match (Lowest MSE)
        best_idx = np.argmin(mse_scores)
        best_train_img = train_images[best_idx]
        best_mse = mse_scores[best_idx]
        
        high_similarity_pairs.append({
            'test_img': test_img_np,
            'train_img': best_train_img,
            'mse': best_mse
        })
        
        count += 1
        print(f"Processed {count}/{num_test_samples} - Min MSE: {best_mse:.6f}", end='\r')
        
    # Sort by MSE ascending (lowest error = most similar first)
    high_similarity_pairs.sort(key=lambda x: x['mse'])
    
    # 3. Visualize Top 5 Matches
    print("\n\nTop 5 Most Similar Pairs (Potential Leakage):")
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))
    
    for i in range(5):
        pair = high_similarity_pairs[i]
        print(f"Pair {i+1}: MSE = {pair['mse']:.6f}")
        
        ax_test = axes[i, 0]
        ax_train = axes[i, 1]
        
        ax_test.imshow(pair['test_img'], cmap='gray')
        ax_test.set_title(f"Test Image\n(MSE: {pair['mse']:.4f})")
        ax_test.axis('off')
        
        ax_train.imshow(pair['train_img'], cmap='gray')
        ax_train.set_title("Closest Train Image")
        ax_train.axis('off')
        
    plt.tight_layout()
    os.makedirs('../reports', exist_ok=True)
    save_path = '../reports/leakage_check.png'
    plt.savefig(save_path)
    print(f"\nSaved visualization to {save_path}")
    
    # Interpretation
    avg_top5_mse = sum(p['mse'] for p in high_similarity_pairs[:5]) / 5
    print("\n--- ANALYSIS REPORT ---")
    if avg_top5_mse < 0.01:
        print("CRITICAL WARNING: High visual similarity detected!")
        print("Found test images that are nearly identical to training images (MSE < 0.01).")
        print("This strongly suggests adjacent MRI slices from the same patient are split across sets.")
        print("Data Leakage is highly likely.")
    elif avg_top5_mse < 0.05:
        print("WARNING: Moderate similarity detected.")
        print("Some images look similar, but might be different patients with similar anatomy.")
    else:
        print("GOOD NEWS: Low similarity detected.")
        print("Test images look distinct from training images.")

if __name__ == "__main__":
    check_leakage()
