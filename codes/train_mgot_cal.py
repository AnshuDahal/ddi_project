"""
train_mgot_cal.py — EfficientNet-B3 + MGOT Fairness Regulariser + Temperature Calibration
===========================================================================================
Usage:
    python train_mgot_cal.py \
        --data_dir  data/ddi \
        --labels    data/ddi_metadata.csv \
        --output    checkpoints/ \
        --epochs    30 \
        --batch     16 \
        --seed      42

Pipeline
--------
1. Load EfficientNet-B3 (ISIC-pretrained or ImageNet if unavailable)
2. Fine-tune on DDI with MGOT fairness regulariser
3. Apply per-group temperature scaling on validation set
4. Save checkpoint + calibration temperatures
5. Save all per-split fairness metrics to CSV

Expected CSV columns (ddi_metadata.csv):
    image_path, malignant (0/1), fitzpatrick (1-6 or I-II/III-IV/V-VI)
"""

import os, argparse, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class DDIDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_path'])
        img      = Image.open(img_path).convert('RGB')
        img      = self.transform(img)
        label    = int(row['malignant'])
        fst      = int(row.get('fst_group', 0))  # 0=FST I-II, 1=FST III-IV, 2=FST V-VI
        return img, label, fst


# ── MGOT Regulariser ──────────────────────────────────────────────────────────

class MGOTRegulariser(nn.Module):
    """
    Simplified MGOT fairness regulariser:
    Penalises the mean L2 distance between group-level feature centroids.
    Full MGOT uses optimal transport; this is the contrastive approximation.
    """
    def __init__(self, n_groups=3, lambda_reg=0.5):
        super().__init__()
        self.n_groups   = n_groups
        self.lambda_reg = lambda_reg

    def forward(self, features, fst_groups):
        centroids = []
        for g in range(self.n_groups):
            mask = (fst_groups == g)
            if mask.sum() > 0:
                centroids.append(features[mask].mean(0))

        if len(centroids) < 2:
            return torch.tensor(0.0, device=features.device)

        loss = torch.tensor(0.0, device=features.device)
        count = 0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                loss += torch.norm(centroids[i] - centroids[j], p=2)
                count += 1
        return self.lambda_reg * loss / max(count, 1)


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, n_iter=100):
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=n_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval_step)
        return self.temperature.item()


def calibrate_per_group(model, val_loader, device, n_groups=3):
    model.eval()
    all_logits, all_labels, all_fst = [], [], []

    with torch.no_grad():
        for imgs, labels, fst in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_fst.append(fst)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_fst    = torch.cat(all_fst)

    temperatures = {}
    for g in range(n_groups):
        mask = (all_fst == g)
        if mask.sum() < 10:
            temperatures[g] = 1.0
            continue
        scaler = TemperatureScaler()
        t = scaler.fit(all_logits[mask], all_labels[mask])
        temperatures[g] = t
        print(f"  FST group {g}: T = {t:.4f}")

    return temperatures


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, mgot, device, feature_hook):
    model.train()
    total_loss = 0
    for imgs, labels, fst in loader:
        imgs, labels, fst = imgs.to(device), labels.to(device), fst.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        cls_loss  = criterion(logits, labels)
        fair_loss = mgot(feature_hook['features'], fst)
        loss      = cls_loss + fair_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, labels_all, fst_all = [], [], []
    with torch.no_grad():
        for imgs, labels, fst in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1]
            preds.extend(probs.cpu().numpy())
            labels_all.extend(labels.numpy())
            fst_all.extend(fst.numpy())

    df = pd.DataFrame({'prob': preds, 'label': labels_all, 'fst': fst_all})
    results = {}
    for g in df['fst'].unique():
        sub = df[df['fst'] == g]
        if sub['label'].nunique() < 2:
            continue
        auc = roc_auc_score(sub['label'], sub['prob'])
        preds_bin = (sub['prob'] >= 0.5).astype(int)
        fn_rate = ((preds_bin == 0) & (sub['label'] == 1)).sum() / max((sub['label'] == 1).sum(), 1)
        results[g] = {'AUC': round(auc, 4), 'FN_Rate': round(float(fn_rate), 4)}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  default='data/ddi')
    parser.add_argument('--labels',    default='data/ddi_metadata.csv')
    parser.add_argument('--output',    default='checkpoints')
    parser.add_argument('--epochs',    type=int,   default=30)
    parser.add_argument('--batch',     type=int,   default=16)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--lambda_reg',type=float, default=0.5)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    if not TIMM_AVAILABLE:
        print("ERROR: timm not installed. Run: pip install timm")
        return

    os.makedirs(args.output, exist_ok=True)
    os.makedirs('results',   exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = pd.read_csv(args.labels)
    # Normalise FST to 0/1/2
    fst_map = {'I': 0, 'II': 0, 'III': 1, 'IV': 1, 'V': 2, 'VI': 2,
               1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    df['fst_group'] = df['fitzpatrick'].map(fst_map).fillna(0).astype(int)

    # 10-fold MC-CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    all_fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(df, df['malignant'])):
        print(f"\n── Fold {fold+1}/10 ──")
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[test_idx[:len(test_idx)//2]]
        test_df  = df.iloc[test_idx[len(test_idx)//2:]]

        train_ds = DDIDataset(train_df, args.data_dir)
        val_ds   = DDIDataset(val_df,   args.data_dir)
        test_ds  = DDIDataset(test_df,  args.data_dir)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=2)

        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2).to(device)

        # Feature hook for MGOT
        feature_hook = {}
        def hook_fn(m, inp, out):
            feature_hook['features'] = out.flatten(1)
        model.global_pool.register_forward_hook(hook_fn)

        mgot      = MGOTRegulariser(lambda_reg=args.lambda_reg)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, mgot, device, feature_hook)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")

        # Calibration
        print("  Calibrating per FST group...")
        temperatures = calibrate_per_group(model, val_loader, device)

        # Evaluate
        fold_results = evaluate(model, test_loader, device)
        for g, metrics in fold_results.items():
            all_fold_results.append({
                'fold': fold+1, 'fst_group': g,
                'AUC': metrics['AUC'], 'FN_Rate': metrics['FN_Rate'],
                'T_cal': temperatures.get(g, 1.0),
            })

        # Save checkpoint for last fold
        if fold == 9:
            torch.save({
                'model_state_dict': model.state_dict(),
                'temperatures':     temperatures,
                'fold':             fold + 1,
            }, os.path.join(args.output, 'mgot_cal.pth'))
            print(f"  ✓ Checkpoint saved.")

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('results/mccv_fold_results.csv', index=False)
    summary = results_df.groupby('fst_group').agg({'AUC': ['mean','std'], 'FN_Rate': ['mean','std']})
    summary.to_csv('results/mccv_summary.csv')
    print("\n✓ MC-CV complete. Results saved to results/")
    print(summary)


if __name__ == '__main__':
    main()
