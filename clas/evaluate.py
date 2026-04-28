"""
Evaluate trained models and produce comparison report.
Usage:
    python evaluate.py --data_dir /path/to/dvm --results_dir checkpoints
"""
import argparse
import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_from_folder, load_from_csv, get_loaders, COLOR_CLASSES
from models import get_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def plot_training_curves(results_list, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res in results_list:
        name = res['model']
        epochs = [h['epoch'] for h in res['history']]
        val_f1 = [h['val_f1'] for h in res['history']]
        val_loss = [h['val_loss'] for h in res['history']]
        axes[0].plot(epochs, val_f1, label=name, marker='o', markersize=3)
        axes[1].plot(epochs, val_loss, label=name, marker='o', markersize=3)

    axes[0].axhline(0.8, color='red', linestyle='--', label='Target F1=0.8')
    axes[0].set_title('Validation F1 Macro')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('F1 Macro')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {save_path}")


def plot_comparison_bar(results_list, save_path):
    names = [r['model'] for r in results_list]
    f1s = [r['test_f1_macro'] for r in results_list]
    accs = [r['test_acc'] for r in results_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1s, width, label='F1 Macro', color='steelblue')
    bars2 = ax.bar(x + width / 2, accs, width, label='Accuracy', color='darkorange')

    ax.axhline(0.8, color='red', linestyle='--', label='Target F1=0.8')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Test Set')
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison bar chart: {save_path}")


def evaluate_all(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.csv:
        splits = load_from_csv(args.csv, root_dir=args.data_dir, test_size=0.2, val_size=0.1)
    else:
        splits = load_from_folder(args.data_dir, test_size=0.2, val_size=0.1)

    _, _, test_samples = splits
    _, _, test_loader = get_loaders(
        test_samples, test_samples, test_samples,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model_names = ['resnet18_scratch', 'resnet50_pretrained', 'efficientnet_pretrained']
    results_list = []
    os.makedirs(args.results_dir, exist_ok=True)

    for name in model_names:
        ckpt = os.path.join(args.results_dir, f'{name}_best.pt')
        json_path = os.path.join(args.results_dir, f'{name}_result.json')

        if not os.path.exists(ckpt):
            print(f"Checkpoint not found for {name}, skipping.")
            continue

        print(f"\n--- Evaluating {name} ---")
        model = get_model(name, num_classes=len(COLOR_CLASSES)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds, labels = predict(model, test_loader, device)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Macro : {f1:.4f}")
        print(classification_report(labels, preds, target_names=COLOR_CLASSES, zero_division=0))

        plot_confusion_matrix(
            labels, preds, COLOR_CLASSES,
            title=f'Confusion Matrix — {name}',
            save_path=os.path.join(args.results_dir, f'{name}_cm.png')
        )

        # Load history from json if available
        if os.path.exists(json_path):
            with open(json_path) as f:
                res = json.load(f)
            res['test_f1_macro'] = f1
            res['test_acc'] = acc
        else:
            res = {'model': name, 'test_f1_macro': f1, 'test_acc': acc, 'history': []}

        results_list.append(res)

    if len(results_list) > 1:
        plot_comparison_bar(
            results_list,
            save_path=os.path.join(args.results_dir, 'comparison.png')
        )

    histories = [r for r in results_list if r.get('history')]
    if histories:
        plot_training_curves(
            histories,
            save_path=os.path.join(args.results_dir, 'training_curves.png')
        )

    print("\n=== Final Comparison ===")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1 Macro':>10}")
    print("-" * 52)
    best = None
    for r in results_list:
        marker = ' <-- BEST' if best is None or r['test_f1_macro'] > best['test_f1_macro'] else ''
        if not best or r['test_f1_macro'] > best['test_f1_macro']:
            best = r
        print(f"{r['model']:<30} {r['test_acc']:>10.4f} {r['test_f1_macro']:>10.4f}{marker}")

    if best:
        passed = best['test_f1_macro'] >= 0.8
        status = "PASSED (>= 0.8)" if passed else "FAILED (< 0.8)"
        print(f"\nBest model: {best['model']} | F1_macro={best['test_f1_macro']:.4f} | {status}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--csv', type=str, default=None)
    p.add_argument('--results_dir', type=str, default='checkpoints')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_all(args)
