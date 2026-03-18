import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import Config, BirdsongDataset, build_cache
from model import BirdsongCNN


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalise_spectrogram(batch: torch.Tensor) -> torch.Tensor:
    """Per-sample z-normalisation (mean 0, std 1) over the spectrogram."""
    mean = batch.mean(dim=(2, 3), keepdim=True)
    std  = batch.std(dim=(2, 3), keepdim=True) + 1e-6
    return (batch - mean) / std


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for specs, labels in loader:
        specs  = normalise_spectrogram(specs).to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    for specs, labels in loader:
        specs  = normalise_spectrogram(specs).to(device)
        labels = labels.to(device)
        logits = model(specs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load CSV ---
    # Expected columns: filename (e.g. "XC12345.flac"), label (e.g. "Robin")
    df = pd.read_csv(cfg.CSV_PATH)
    print(f"Dataset: {len(df)} clips, {df['label'].nunique()} species")

    # --- Encode labels ---
    le = LabelEncoder()
    le.fit(df["label"])
    n_classes = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    # Save encoder for inference later
    Path(cfg.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{cfg.MODEL_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # --- Build spectrogram cache ---
    print("Building spectrogram cache …")
    build_cache(cfg, le, df)

    # --- Train / val / test split (stratified) ---
    train_df, test_df = train_test_split(
        df, test_size=cfg.TEST_SPLIT, stratify=df["label"], random_state=cfg.SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=cfg.VAL_SPLIT / (1 - cfg.TEST_SPLIT),
        stratify=train_df["label"], random_state=cfg.SEED
    )
    print(f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # --- Compute class weights to handle imbalanced datasets ---
    counts = train_df["label"].value_counts()
    weights = torch.tensor(
        [1.0 / counts[cls] for cls in le.classes_], dtype=torch.float32
    ).to(device)

    # --- Dataloaders ---
    train_ds = BirdsongDataset(train_df, le, cfg, augment=True)
    val_ds   = BirdsongDataset(val_df,   le, cfg, augment=False)
    test_ds  = BirdsongDataset(test_df,  le, cfg, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, optimiser, scheduler ---
    model     = BirdsongCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.N_EPOCHS
    )

    # --- Training ---
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.N_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimiser, criterion, device)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.3f} | "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), f"{cfg.MODEL_DIR}/best_model.pt")
            print(f"  ✓ Saved (val acc {best_val_acc:.3f})")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(f"{cfg.MODEL_DIR}/best_model.pt"))
    _, test_acc, preds, labels = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(classification_report(labels, preds, target_names=le.classes_))

    # --- Confusion matrix ---
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(8, n_classes), max(6, n_classes - 2)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix — test set")
    plt.tight_layout()
    plt.savefig(f"{cfg.MODEL_DIR}/confusion_matrix.png", dpi=150)
    print(f"Confusion matrix saved.")

    # --- Training curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"],   label="val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"],   label="val")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.MODEL_DIR}/training_curves.png", dpi=150)


# ── Inference on a new file ────────────────────────────────────────────────────

def predict(flac_path: str, model_dir: str = "models") -> dict:
    """Return the predicted species and class probabilities for one file."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model = BirdsongCNN(n_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt",
                                     map_location=device))
    model.eval()

    from dataset import load_clip, to_log_mel
    y       = load_clip(flac_path, cfg)
    log_mel = torch.tensor(to_log_mel(y, cfg)).unsqueeze(0).to(device)
    log_mel = normalise_spectrogram(log_mel)

    with torch.no_grad():
        probs = torch.softmax(model(log_mel), dim=1).squeeze().cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    return {
        "predicted_species": le.classes_[probs.argmax()],
        "confidence": float(probs.max()),
        "top_5": [
            {"species": le.classes_[i], "probability": float(probs[i])}
            for i in top5_idx
        ],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import json
        result = predict(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        main()