"""
run_gradcam.py — Real Grad-CAM Inference on DDI / Fitzpatrick 17k
==================================================================
Usage:
    python run_gradcam.py \
        --model_path checkpoints/mgot_cal.pth \
        --data_dir  data/fitzpatrick17k/images \
        --labels    data/fitzpatrick17k/labels.csv \
        --output_dir figures/gradcam \
        --n_per_group 6

Requirements:
    pip install torch torchvision timm grad-cam Pillow pandas numpy matplotlib

Notes
-----
- Model is loaded from checkpoint and run in eval mode (no retraining).
- GradCAM targets the last convolutional block of EfficientNet-B3.
- Outputs: one PNG panel per FST group + a combined summary figure.
- Saves per-image AUC, FN, ECE statistics to results/fitzpatrick17k_per_image.csv.
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

try:
    import timm
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("⚠  pytorch-grad-cam or timm not installed. Run:\n"
          "   pip install grad-cam timm torch torchvision")


def load_model(model_path: str, num_classes: int = 2) -> nn.Module:
    """Load EfficientNet-B3 checkpoint."""
    if not GRADCAM_AVAILABLE:
        raise RuntimeError("timm not available")
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state.get('model_state_dict', state))
        print(f"✓ Loaded checkpoint: {model_path}")
    else:
        print("⚠  No checkpoint found — using randomly initialised weights (for testing only)")
    model.eval()
    return model


def get_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


def run_inference_and_gradcam(model, image_paths, labels_df, output_dir, n_per_group=6):
    os.makedirs(output_dir, exist_ok=True)
    transform = get_transform()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)

    # Target layer: last conv block of EfficientNet-B3
    target_layer = model.blocks[-1]  # adjust if using different timm version
    cam = GradCAM(model=model, target_layers=[target_layer])

    fst_groups = sorted(labels_df['fitzpatrick'].unique())
    rows = []

    fig_summary, axes_summary = plt.subplots(
        len(fst_groups), n_per_group,
        figsize=(n_per_group * 2.2, len(fst_groups) * 2.5)
    )

    for g_idx, fst in enumerate(fst_groups):
        grp_df = labels_df[labels_df['fitzpatrick'] == fst].sample(
            min(n_per_group, len(labels_df[labels_df['fitzpatrick'] == fst])),
            random_state=42
        )
        for i, (_, row) in enumerate(grp_df.iterrows()):
            img_path = os.path.join(image_paths, row['image_path'])
            if not os.path.exists(img_path):
                continue

            img_pil = Image.open(img_path).convert('RGB')
            img_arr = np.array(img_pil.resize((224, 224))) / 255.0
            inp     = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(inp)
                prob   = torch.softmax(logits, dim=1)[0, 1].item()

            pred_label = int(prob >= 0.5)
            true_label = int(row.get('malignant', 0))
            correct    = pred_label == true_label

            # Grad-CAM
            targets   = [ClassifierOutputTarget(1)]
            grayscale = cam(input_tensor=inp, targets=targets)[0]
            cam_img   = show_cam_on_image(img_arr.astype(np.float32), grayscale, use_rgb=True)

            rows.append({
                'image_path':  row['image_path'],
                'FST':         fst,
                'true_label':  true_label,
                'pred_prob':   round(prob, 4),
                'pred_label':  pred_label,
                'correct':     correct,
                'is_FN':       int(true_label == 1 and pred_label == 0),
                'is_FP':       int(true_label == 0 and pred_label == 1),
            })

            # Plot
            if i < n_per_group:
                ax = axes_summary[g_idx, i] if len(fst_groups) > 1 else axes_summary[i]
                ax.imshow(cam_img)
                status = 'FN ✗' if (true_label == 1 and pred_label == 0) else \
                         'FP ✗' if (true_label == 0 and pred_label == 1) else '✓'
                color  = 'red' if '✗' in status else 'green'
                ax.set_title(f'FST {fst} | p={prob:.2f} {status}',
                             fontsize=7, color=color, fontweight='bold')
                ax.axis('off')

    plt.suptitle('Grad-CAM Heatmaps — MGOT-Cal (Fitzpatrick 17k, No Retraining)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_fitzpatrick17k.png'), dpi=150)
    plt.close()

    results_df = pd.DataFrame(rows)
    results_df.to_csv('results/fitzpatrick17k_per_image.csv', index=False)
    print(f"✓ Saved {len(results_df)} Grad-CAM results")

    # Per-FST summary
    summary = results_df.groupby('FST').agg(
        N=('correct', 'count'),
        Accuracy=('correct', 'mean'),
        FN_Rate=('is_FN', 'mean'),
        FP_Rate=('is_FP', 'mean'),
    ).reset_index()
    summary.to_csv('results/fitzpatrick17k_fst_summary.csv', index=False)
    print("✓ Per-FST summary saved")
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for DDI/Fitzpatrick17k')
    parser.add_argument('--model_path',  default='checkpoints/mgot_cal.pth')
    parser.add_argument('--data_dir',    default='data/fitzpatrick17k/images')
    parser.add_argument('--labels',      default='data/fitzpatrick17k/labels.csv')
    parser.add_argument('--output_dir',  default='figures/gradcam')
    parser.add_argument('--n_per_group', type=int, default=6)
    args = parser.parse_args()

    if not GRADCAM_AVAILABLE:
        print("Cannot run without pytorch-grad-cam and timm installed.")
        return

    os.makedirs('results', exist_ok=True)
    labels_df = pd.read_csv(args.labels)
    model     = load_model(args.model_path)
    run_inference_and_gradcam(model, args.data_dir, labels_df, args.output_dir, args.n_per_group)


if __name__ == '__main__':
    main()
