import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")


def save_history(history, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'training_history.json')
    
    # DataFrame'e çevirip kaydetmek daha okunabilir olur
    hist_df = pd.DataFrame(history)
    
    hist_df.to_json(filepath, orient='records', indent=4)
        
    print(f"✅ Training history saved to {filepath}")

def plot_training_curves(history, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'training_curves.png')
    
    # DataFrame
    hist_df = pd.DataFrame(history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. Loss Graphs
    ax1.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(hist_df['epoch'], hist_df['val_loss'], label='Validation Loss', color='orange', marker='x')
    ax1.set_title('Loss vs. Epochs', fontsize=16)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 2. Accuracy Graphs
    ax2.plot(hist_df['epoch'], hist_df['train_acc'], label='Train Accuracy', color='blue', marker='o')
    ax2.plot(hist_df['epoch'], hist_df['val_acc'], label='Validation Accuracy', color='orange', marker='x')
    ax2.set_title('Accuracy vs. Epochs', fontsize=16)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    fig.suptitle('Training & Validation Curves', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"✅ Training curves plot saved to {filepath}")
