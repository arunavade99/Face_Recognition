"""
Build an embeddings database for FaceNet with optional fine-tuning.

Workflow:
1) Reads person-wise images from an aligned dataset folder (default: ./aligned).
2) (Optional, default ON) Fine-tunes FaceNet (softmax head) on your classes.
3) Removes the classifier head and extracts L2-normalized embeddings.
4) Saves a compressed database 'embeddings.npz' and a 'labels.txt' mapping.

Usage:
  python build_db.py --aligned_dir aligned --epochs 5 --batch_size 32 --lr 1e-3
  Disable fine-tune:
  python build_db.py --no-finetune

Outputs:
  - embeddings.npz  (keys: person names → mean embedding vectors)
  - labels.txt      (one class name per line, in alphabetical order)
  - facenet_finetuned.pth (if fine-tuned)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1

# finetune and utils

def set_seed(seed: int = 42):
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def save_labels(labels: List[str], path: Path):
    path.write_text("\n".join(labels), encoding="utf-8")

#Training (optional) 

def finetune_facenet(
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
) -> InceptionResnetV1:
    """
    Fine-tunes FaceNet with a softmax head on your dataset.
    Returns the trained classification model (with logits head).
    """
    # Load FaceNet with classifier head
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
    model = model.to(device)

    # Optionally freeze backbone (speeds up + stabilizes on small datasets)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("logits"):  # keep classifier trainable
                p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # returns logits when classify=True
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        print(f"[Finetune] Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {acc:.2f}%")

    return model.eval()


def strip_classifier_to_embeddings(trained_cls_model: InceptionResnetV1, device: torch.device) -> InceptionResnetV1:
    """
    Create an embedding-only FaceNet and load weights from a fine-tuned classifier model.
    Classifier head weights are ignored automatically (strict=False).
    """
    emb_model = InceptionResnetV1(pretrained=None, classify=False)
    # Load everything except 'logits.*'
    emb_model.load_state_dict(trained_cls_model.state_dict(), strict=False)
    return emb_model.eval().to(device)

# --------------- Embedding DB ---------------

@torch.no_grad()
def build_embeddings_db(
    dataset: datasets.ImageFolder,
    device: torch.device,
    batch_size: int,
    emb_model: InceptionResnetV1,
) -> Dict[str, np.ndarray]:
    """
    Passes all images through the embedding model, averages per class,
    and returns {class_name: mean_embedding}.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    num_classes = len(dataset.classes)
    feat_dim = 512  # InceptionResnetV1 default embedding size
    sums = np.zeros((num_classes, feat_dim), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int32)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        embs = emb_model(imgs)  # [B, 512]
        embs = embs.cpu().numpy()
        embs = l2_normalize(embs, axis=1)

        labels_np = labels.numpy()
        for i, cls in enumerate(labels_np):
            sums[cls] += embs[i]
            counts[cls] += 1

    means = np.zeros_like(sums)
    for cls_idx in range(num_classes):
        if counts[cls_idx] > 0:
            means[cls_idx] = l2_normalize(sums[cls_idx] / counts[cls_idx])
        else:
            means[cls_idx] = np.zeros((feat_dim,), dtype=np.float32)

    db = {dataset.classes[i]: means[i] for i in range(num_classes) if counts[i] > 0}
    return db

# main function

def main():
    parser = argparse.ArgumentParser(description="Build FaceNet embeddings DB (with optional fine-tuning).")
    parser.add_argument("--aligned_dir", type=str, default="aligned", help="Path to aligned dataset root.")
    parser.add_argument("--out_npz", type=str, default="embeddings.npz", help="Output path for embeddings DB.")
    parser.add_argument("--labels_out", type=str, default="labels.txt", help="Where to save label names.")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for both train and embedding passes.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for fine-tuning.")
    parser.add_argument("--no-finetune", action="store_true", help="Disable fine-tuning (use pretrained embeddings).")
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze backbone during finetune (slower, more precise).")
    args = parser.parse_args()

    set_seed(42)

    aligned_dir = Path(args.aligned_dir)
    if not aligned_dir.exists():
        raise FileNotFoundError(f"Aligned dataset not found: {aligned_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms: FaceNet expects inputs in [-1, 1], so normalize with mean=0.5, std=0.5
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # broadcast to 3 channels
    ])

    # Load dataset
    dataset = datasets.ImageFolder(str(aligned_dir), transform=transform)
    if len(dataset.classes) == 0:
        raise RuntimeError("No classes found in aligned dataset.")
    print(f"Found {len(dataset.classes)} classes: {dataset.classes}")

    # Optional mild augmentation ONLY for finetuning
    aug_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    aug_dataset = datasets.ImageFolder(str(aligned_dir), transform=aug_transform)

    # Fine-tune or not
    if args.no_finetune:
        print("Finetune: OFF → using pretrained FaceNet embeddings.")
        emb_model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
    else:
        print("Finetune: ON")
        train_loader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        cls_model = finetune_facenet(
            train_loader=train_loader,
            num_classes=len(dataset.classes),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=not args.unfreeze,
        )
        # Save fine-tuned weights (classifier version)
        torch.save(cls_model.state_dict(), "facenet_finetuned.pth")
        print("Saved fine-tuned model: facenet_finetuned.pth")

        # Strip classifier to get pure embedding model
        emb_model = strip_classifier_to_embeddings(cls_model, device)

    # Build embeddings DB (average per person)
    print("Building embeddings database...")
    db = build_embeddings_db(dataset, device, args.batch_size, emb_model)

    # Save DB (compressed) + labels
    if len(db) == 0:
        raise RuntimeError("No embeddings computed. Check your aligned images.")
    np.savez_compressed(args.out_npz, **db)
    save_labels(dataset.classes, Path(args.labels_out))
    print(f"✅ Saved embeddings DB → {args.out_npz}")
    print(f"✅ Saved labels → {args.labels_out}")
    print("Done.")

if __name__ == "__main__":
    main()
