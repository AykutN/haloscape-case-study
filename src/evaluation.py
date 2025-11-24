import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _gather_predictions(model, dataloader, device):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return all_labels, all_preds


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, split_name: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix - {split_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_split(model, dataloader, device, class_names: List[str], split_name: str, save_dir: str) -> Dict[str, float]:
    os.makedirs(save_dir, exist_ok=True)
    labels, preds = _gather_predictions(model, dataloader, device)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)

    split_dir = os.path.join(save_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }

    with open(os.path.join(split_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(split_dir, 'classification_report.csv'))

    _plot_confusion_matrix(cm, class_names, os.path.join(split_dir, 'confusion_matrix.png'), split_name)

    print(f"\n{split_name.upper()} METRICS")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}")

    return metrics
