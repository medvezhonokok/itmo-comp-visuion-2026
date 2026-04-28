"""
Training pipeline for car color classification.
Usage:
    python train.py --data_dir /path/to/dvm --model resnet18_scratch --epochs 30
    python train.py --data_dir /path/to/dvm --model resnet50_pretrained --epochs 15
    python train.py --data_dir /path/to/dvm --model efficientnet_pretrained --epochs 15
"""
import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from dataset import load_from_folder, load_from_csv, get_loaders, COLOR_CLASSES
from models import get_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    if args.csv:
        splits = load_from_csv(args.csv, root_dir=args.data_dir,
                               test_size=0.2, val_size=0.1)
    else:
        splits = load_from_folder(args.data_dir, test_size=0.2, val_size=0.1)

    train_samples, val_samples, test_samples = splits
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_loader, val_loader, test_loader = get_loaders(
        train_samples, val_samples, test_samples,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    num_classes = len(COLOR_CLASSES)
    model = get_model(args.model, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Different LR for backbone vs head when fine-tuning
    if args.model in ('resnet50_pretrained', 'efficientnet_pretrained'):
        if args.model == 'resnet50_pretrained':
            backbone_params = list(model.parameters())[:-2]
            head_params = list(model.fc.parameters())
        else:
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
            head_params = list(model.classifier.parameters())

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = os.path.join(args.save_dir, f'{args.model}_best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best model (val_f1={best_f1:.4f})")

    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'{args.model}_best.pt'),
                                     map_location=device))
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, device)
    print(f"\n=== Test results for {args.model} ===")
    print(f"  Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1_macro={test_f1:.4f}")

    result = {
        'model': args.model,
        'best_val_f1': best_f1,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1_macro': test_f1,
        'history': history,
    }
    result_path = os.path.join(args.save_dir, f'{args.model}_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description='Car color classifier training')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to DVM dataset root (ImageFolder structure or use --csv)')
    p.add_argument('--csv', type=str, default=None,
                   help='Optional CSV file with [image_path, color] columns')
    p.add_argument('--model', type=str, default='resnet18_scratch',
                   choices=['resnet18_scratch', 'resnet50_pretrained', 'efficientnet_pretrained'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
