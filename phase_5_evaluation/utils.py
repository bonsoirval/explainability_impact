import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_fscore_support

# ── Shared colour palette (matches plot_from_csv.py) ─────────────────────────
METHOD_COLORS = {
    "Saliency": "#3A86FF",
    "Grad-CAM": "#FF006E",
    "LIME":     "#8338EC",
    "SHAP":     "#FFBE0B",
}
FALLBACK_COLORS = ["#3A86FF", "#FF006E", "#8338EC", "#FFBE0B",
                   "#06D6A0", "#FB5607", "#023E8A", "#E63946"]


def _method_color(method: str, idx: int = 0) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _save(fig, save_path: str, dpi: int = 200) -> None:
    """Save figure and close it."""
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Existing helpers
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    t = tensor.clone().squeeze()
    if t.is_cuda:
        t = t.cpu()
    for channel, m, s in zip(t, mean, std):
        channel.mul_(s).add_(m)
    np_img = t.permute(1, 2, 0).numpy()
    np_img = np.clip(np_img, 0, 1)
    return np_img


def plot_explanations(original_tensor, saliency_map, gradcam_map, lime_img, shap_map, save_path=None):
    raw_img = denormalize(original_tensor)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(raw_img)
    axes[0].set_title("Original Leaf")
    axes[0].axis('off')
    
    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title("Vanilla Saliency")
    axes[1].axis('off')
    
    axes[2].imshow(raw_img)
    axes[2].imshow(gradcam_map, cmap='jet', alpha=0.5)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis('off')
    
    axes[3].imshow(lime_img)
    axes[3].set_title("LIME Highlights")
    axes[3].axis('off')
    
    axes[4].imshow(shap_map, cmap='RdBu_r', vmin=-np.max(np.abs(shap_map)), vmax=np.max(np.abs(shap_map)))
    axes[4].set_title("SHAP Attributions")
    axes[4].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_benchmark_distributions(raw_df, save_path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    methods = raw_df['Method'].unique()
    has_vram = (raw_df['Peak VRAM (MB)'].sum() > 0)
    num_plots = 3 if has_vram else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5.5))
    if num_plots == 1:
        axes = [axes]
        
    colors = ['#3A86FF', '#FF006E', '#8338EC', '#FFBE0B']
    box_props = dict(patch_artist=True, medianprops=dict(color='black', linewidth=1.5), flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    ax_lat = axes[0]
    data_lat = [raw_df[raw_df['Method'] == m]['Latency (s)'].values for m in methods]
    bp_lat = ax_lat.boxplot(data_lat, tick_labels=list(methods), **box_props)
    
    for patch, color in zip(bp_lat['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_edgecolor(color)
        
    for i, m in enumerate(methods):
        y = raw_df[raw_df['Method'] == m]['Latency (s)'].values
        x = np.random.normal(i + 1, 0.05, size=len(y))
        ax_lat.scatter(x, y, color=colors[i], alpha=0.7, s=25, edgecolors='black', linewidths=0.5)
        
    ax_lat.set_yscale('log')
    ax_lat.set_ylabel('Latency (seconds) - Log Scale', fontsize=11, fontweight='bold')
    ax_lat.set_title('Explainability Generation Latency', fontsize=12, fontweight='bold', pad=10)
    ax_lat.grid(True, which="both", ls="--", alpha=0.3)
    
    ax_ram = axes[1]
    data_ram = [raw_df[raw_df['Method'] == m]['Peak RAM (MB)'].values for m in methods]
    bp_ram = ax_ram.boxplot(data_ram, tick_labels=list(methods), **box_props)
    
    for patch, color in zip(bp_ram['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_edgecolor(color)
        
    for i, m in enumerate(methods):
        y = raw_df[raw_df['Method'] == m]['Peak RAM (MB)'].values
        x = np.random.normal(i + 1, 0.05, size=len(y))
        ax_ram.scatter(x, y, color=colors[i], alpha=0.7, s=25, edgecolors='black', linewidths=0.5)
        
    ax_ram.set_ylabel('Peak RAM Usage (MB)', fontsize=11, fontweight='bold')
    ax_ram.set_title('Peak RAM Allocation', fontsize=12, fontweight='bold', pad=10)
    ax_ram.grid(True, which="both", ls="--", alpha=0.3)
    
    if has_vram:
        ax_vram = axes[2]
        data_vram = [raw_df[raw_df['Method'] == m]['Peak VRAM (MB)'].values for m in methods]
        bp_vram = ax_vram.boxplot(data_vram, tick_labels=list(methods), **box_props)
        
        for patch, color in zip(bp_vram['boxes'], colors[:len(methods)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
            patch.set_edgecolor(color)
            
        for i, m in enumerate(methods):
            y = raw_df[raw_df['Method'] == m]['Peak VRAM (MB)'].values
            x = np.random.normal(i + 1, 0.05, size=len(y))
            ax_vram.scatter(x, y, color=colors[i], alpha=0.7, s=25, edgecolors='black', linewidths=0.5)
            
        ax_vram.set_ylabel('Peak VRAM Usage (MB)', fontsize=11, fontweight='bold')
        ax_vram.set_title('Peak VRAM Allocation', fontsize=12, fontweight='bold', pad=10)
        ax_vram.grid(True, which="both", ls="--", alpha=0.3)
        
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
    plt.suptitle(f'Computational Performance Distribution ({len(raw_df) // len(methods)} Runs)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Thesis-quality plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, model_name: str, save_path: str) -> None:
    """
    Plot training and validation loss + accuracy curves over epochs.

    Parameters
    ----------
    history    : Dict returned by train_model() with keys
                 train_loss, train_acc, val_loss, val_acc.
    model_name : Used in the figure title.
    save_path  : Output PNG path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, history["train_loss"], "o-", color="#3A86FF",
                 linewidth=2, markersize=6, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"],   "s--", color="#FF006E",
                 linewidth=2, markersize=6, label="Val Loss")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax_loss.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax_loss.legend(fontsize=11)
    ax_loss.grid(True, ls="--", alpha=0.35)
    ax_loss.spines[["top", "right"]].set_visible(False)
    ax_loss.set_xticks(list(epochs))

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", color="#3A86FF",
                linewidth=2, markersize=6, label="Train Acc")
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", color="#FF006E",
                linewidth=2, markersize=6, label="Val Acc")
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax_acc.legend(fontsize=11)
    ax_acc.grid(True, ls="--", alpha=0.35)
    ax_acc.spines[["top", "right"]].set_visible(False)
    ax_acc.set_xticks(list(epochs))
    ax_acc.set_ylim(0, 105)

    fig.suptitle(f"{model_name.replace('_', ' ').title()} — Learning Curves",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrix(model, val_loader, class_names: list, device, save_path: str) -> None:
    """
    Compute and plot the confusion matrix on the validation set.

    Parameters
    ----------
    model       : Trained PyTorch model (eval mode).
    val_loader  : Validation DataLoader.
    class_names : List of string class labels.
    device      : torch.device.
    save_path   : Output PNG path.
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = torch.argmax(model(inputs), dim=1)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1

    # Normalise rows to percentages for the colour scale
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1           # avoid /0
    cm_norm = cm_norm / row_sums * 100    # row-wise recall %

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            text_color = "white" if cm_norm[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=7,
                    color=text_color, fontweight="bold" if i == j else "normal")

    # Ticks
    short_names = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title("Validation Confusion Matrix\n(row-normalised recall %)",
                 fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=10)

    plt.tight_layout()
    _save(fig, save_path)


def plot_multi_class_explanations(
    suite,
    val_dataset,
    class_names: list,
    device,
    save_path: str,
    max_classes: int = 10,
) -> None:
    """
    Show Original / Saliency / Grad-CAM for one sample from each disease class.

    Columns: Original | Saliency | Grad-CAM
    Rows   : one per class (up to max_classes)

    Parameters
    ----------
    suite       : ExplainabilitySuite instance.
    val_dataset : PyTorch Dataset (not DataLoader) — underlying dataset object.
    class_names : List of string class labels.
    device      : torch.device.
    save_path   : Output PNG path.
    max_classes : Maximum number of classes to display.
    """
    # Build a class → first matching sample index map
    n_classes = min(len(class_names), max_classes)
    class_sample = {}
    for idx in range(len(val_dataset)):
        _, label = val_dataset[idx]
        if label not in class_sample:
            class_sample[label] = idx
        if len(class_sample) == n_classes:
            break

    selected = sorted(class_sample.items())   # [(label_idx, sample_idx), ...]

    fig, axes = plt.subplots(
        n_classes, 3,
        figsize=(9, 3.2 * n_classes),
        squeeze=False,
    )

    col_titles = ["Original Leaf", "Vanilla Saliency", "Grad-CAM Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    suite.model.eval()
    for row, (label_idx, sample_idx) in enumerate(selected):
        img_tensor, _ = val_dataset[sample_idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # ── Original ──────────────────────────────────────────────────────────
        raw_img = denormalize(img_tensor)
        axes[row][0].imshow(raw_img)
        axes[row][0].axis("off")

        # ── Saliency ──────────────────────────────────────────────────────────
        try:
            sal = suite.explain_saliency(input_tensor, label_idx)
            axes[row][1].imshow(sal, cmap="hot")
        except Exception:
            axes[row][1].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row][1].transAxes)
        axes[row][1].axis("off")

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        try:
            gcam = suite.explain_gradcam(input_tensor, label_idx)
            axes[row][2].imshow(raw_img)
            axes[row][2].imshow(gcam, cmap="jet", alpha=0.5)
        except Exception:
            axes[row][2].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=axes[row][2].transAxes)
        axes[row][2].axis("off")

        # Row label (short class name)
        short = class_names[label_idx].replace("Tomato___", "").replace("_", " ")
        axes[row][0].set_ylabel(short, fontsize=9, rotation=0,
                                labelpad=80, va="center")

    fig.suptitle("XAI Explanations Across Tomato Disease Classes",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    _save(fig, save_path)


def plot_latency_vs_ram(raw_df, model_name: str, save_path: str) -> None:
    """
    Scatter plot of mean Latency (x) vs mean Peak RAM (y) for each XAI method.
    Positions each method in a 2-D computational cost space.

    Parameters
    ----------
    raw_df     : Raw-runs DataFrame from XAIBenchmarker.
    model_name : Used in the figure title.
    save_path  : Output PNG path.
    """
    methods = list(raw_df["Method"].unique())
    means = raw_df.groupby("Method")[["Latency (s)", "Peak RAM (MB)"]].mean()

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, m in enumerate(methods):
        color = _method_color(m, i)
        x = means.loc[m, "Latency (s)"]
        y = means.loc[m, "Peak RAM (MB)"]
        ax.scatter(x, y, s=160, color=color, edgecolors="white",
                   linewidths=1.2, zorder=5, label=m)
        ax.annotate(
            m, (x, y),
            textcoords="offset points", xytext=(8, 4),
            fontsize=10, fontweight="bold", color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean Latency (s) — Log Scale", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Peak RAM (MB)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Latency vs Peak RAM\n"
        "(lower-left = cheapest)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    _save(fig, save_path)


def plot_run_over_run(raw_df, model_name: str, save_path: str) -> None:
    """
    Line chart showing per-run latency for each XAI method.
    Reveals warmup effects and variance across repeated measurements.

    Parameters
    ----------
    raw_df     : Raw-runs DataFrame from XAIBenchmarker.
    model_name : Used in the figure title.
    save_path  : Output PNG path.
    """
    methods = list(raw_df["Method"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, m in enumerate(methods):
        color = _method_color(m, i)
        sub = raw_df[raw_df["Method"] == m].sort_values("Run")
        ax.plot(sub["Run"], sub["Latency (s)"],
                "o-", color=color, linewidth=2, markersize=6,
                label=m, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Run Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latency (s) — Log Scale", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Latency Across Repeated Runs",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, save_path)


def plot_cv_stability(raw_df, model_name: str, save_path: str) -> None:
    """
    Bar chart of Coefficient of Variation (CV = std / mean) of latency per method.
    Lower CV = more reproducible / stable XAI method.

    Parameters
    ----------
    raw_df     : Raw-runs DataFrame from XAIBenchmarker.
    model_name : Used in the figure title.
    save_path  : Output PNG path.
    """
    methods = list(raw_df["Method"].unique())
    cvs = []
    for m in methods:
        vals = raw_df[raw_df["Method"] == m]["Latency (s)"].values
        cv = vals.std() / vals.mean() if vals.mean() > 0 else 0.0
        cvs.append(cv * 100)   # as a percentage

    colors = [_method_color(m, i) for i, m in enumerate(methods)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, cvs, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.55, alpha=0.85)

    # Value labels on top of each bar
    for bar, cv in zip(bars, cvs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{cv:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Coefficient of Variation (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Latency Stability\n"
        "(CV = σ / μ × 100 — lower is more reproducible)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(cvs) * 1.25 + 1)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    _save(fig, save_path)


def plot_confidence_bar(model, input_tensor, class_names: list, device, save_path: str) -> None:
    """
    Horizontal bar chart of the model's softmax output probabilities for all
    classes on a single input image.

    Parameters
    ----------
    model        : Trained PyTorch model (eval mode).
    input_tensor : (1, C, H, W) tensor already on ``device``.
    class_names  : List of string class labels.
    device       : torch.device.
    save_path    : Output PNG path.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    short_names = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]
    pred_idx = int(np.argmax(probs))

    # Sort ascending so highest bar is at top
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_names = [short_names[i] for i in order]

    colors = ["#3A86FF" if i != pred_idx else "#FF006E" for i in order]

    fig, ax = plt.subplots(figsize=(7, max(4, len(class_names) * 0.45)))
    bars = ax.barh(range(len(sorted_names)), sorted_probs * 100,
                   color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)

    for bar, prob in zip(bars, sorted_probs):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Softmax Probability (%)", fontsize=11, fontweight="bold")
    ax.set_title("Model Prediction Confidence\n(red = predicted class)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)



# ─────────────────────────────────────────────────────────────────────────────
# THESIS-QUALITY PLOTS  (Phase 2 → 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_distribution(root_dir: str, save_path: str) -> None:
    """
    Horizontal bar chart of per-class image counts in the dataset.
    Reveals class imbalance — a critical data-understanding check for a thesis.
    """
    class_counts = {}
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = sum(1 for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if n > 0:
            short = cls.replace("Tomato___", "").replace("_", " ")
            class_counts[short] = n

    if not class_counts:
        print(f"  [skip] No images found in {root_dir}")
        return

    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    order  = np.argsort(counts)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    total  = sum(counts)
    cmap   = plt.cm.get_cmap("tab10", len(labels))
    colors = [cmap(i) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.55)))
    bars = ax.barh(range(len(labels)), counts, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.65)
    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_width() + total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({pct:.1f}%)", va="center", ha="left", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Dataset Class Distribution  (total = {total:,} images)\n"
        "PlantVillage — Tomato leaf subset",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(counts) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.axvline(total / len(labels), ls=":", color="grey",
               linewidth=1.2, label="Balanced mean")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    _save(fig, save_path)


def plot_per_class_metrics(model, val_loader, class_names: list,
                            device, save_path: str) -> None:
    """
    Grouped bar chart of Precision, Recall, and F1-score per class.
    Highlights which disease categories the model struggles with — essential
    for a credible evaluation chapter.
    """
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            preds  = torch.argmax(model(inputs), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(class_names))), zero_division=0,
    )
    short_names = [c.replace("Tomato___", "").replace("_", " ")
                   for c in class_names]
    x = np.arange(len(short_names))
    w = 0.27

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.1), 5.5))
    b1 = ax.bar(x - w, precision * 100, w, label="Precision",
                color="#3A86FF", edgecolor="white", linewidth=0.8, alpha=0.88)
    b2 = ax.bar(x,     recall    * 100, w, label="Recall",
                color="#FF006E", edgecolor="white", linewidth=0.8, alpha=0.88)
    b3 = ax.bar(x + w, f1        * 100, w, label="F1-Score",
                color="#8338EC", edgecolor="white", linewidth=0.8, alpha=0.88)

    for group in (b1, b2, b3):
        for bar in group:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("Per-Class Metrics — Precision, Recall & F1-Score",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.axhline(80, ls=":", color="grey", linewidth=1)
    plt.tight_layout()
    _save(fig, save_path)


def plot_violin_latency(raw_df, model_name: str, save_path: str) -> None:
    """
    Violin plot of per-method latency distributions.
    Reveals bimodality and tail behaviour better than a plain box plot.
    """
    methods = list(raw_df["Method"].unique())
    data    = [raw_df[raw_df["Method"] == m]["Latency (s)"].values
               for m in methods]
    colors  = [_method_color(m, i) for i, m in enumerate(methods)]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, positions=range(1, len(methods) + 1),
                          showmedians=True, showextrema=True, widths=0.55)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor(col)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
    for k in ("cmins", "cmaxes", "cbars"):
        parts[k].set_color("grey")

    for i, (m, col) in enumerate(zip(methods, colors)):
        y = raw_df[raw_df["Method"] == m]["Latency (s)"].values
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, color=col, s=22, alpha=0.75,
                   edgecolors="black", linewidths=0.4, zorder=5)

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Latency (s) — Log Scale", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Latency Distribution (Violin)\n"
        "Kernel density + median + individual run points",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", which="both", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, save_path)


def plot_throughput_bar(raw_df, model_name: str, save_path: str) -> None:
    """
    Bar chart of mean inference throughput (samples/second) per XAI method.
    """
    methods = list(raw_df["Method"].unique())
    throughputs, errors = [], []
    for m in methods:
        lats = raw_df[raw_df["Method"] == m]["Latency (s)"].values
        tps  = 1.0 / lats
        throughputs.append(tps.mean())
        errors.append(tps.std())

    colors = [_method_color(m, i) for i, m in enumerate(methods)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, throughputs, color=colors, edgecolor="white",
                  linewidth=1.0, width=0.55, alpha=0.88,
                  yerr=errors, capsize=5,
                  error_kw=dict(ecolor="black", elinewidth=1.2, capthick=1.2))
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(errors) * 0.08,
                f"{tp:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (samples / second)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — XAI Method Throughput\n"
        "(mean ± std  —  higher is faster)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_ylim(0, max(throughputs) * 1.3)
    plt.tight_layout()
    _save(fig, save_path)


def plot_ram_efficiency(raw_df, model_name: str, save_path: str) -> None:
    """
    Bar chart of RAM cost per unit latency (MB·s) — efficiency ratio.
    """
    methods = list(raw_df["Method"].unique())
    ratios, labels_list = [], []
    for m in methods:
        sub  = raw_df[raw_df["Method"] == m]
        eff  = (sub["Peak RAM (MB)"].values * sub["Latency (s)"].values).mean()
        ratios.append(eff); labels_list.append(m)

    colors = [_method_color(m, i) for i, m in enumerate(methods)]
    order  = np.argsort(ratios)
    ratios      = [ratios[i]      for i in order]
    labels_list = [labels_list[i] for i in order]
    colors      = [colors[i]      for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(range(len(labels_list)), ratios, color=colors,
                   edgecolor="white", linewidth=0.8, height=0.55, alpha=0.88)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_width() + max(ratios) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{r:.3f} MB·s", va="center", ha="left",
                fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(labels_list)))
    ax.set_yticklabels(labels_list, fontsize=11)
    ax.set_xlabel("Mean RAM × Latency (MB·s)  —  lower is better",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Computational Efficiency Ratio\n"
        "RAM cost per unit explanation time",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.3)
    ax.set_xlim(0, max(ratios) * 1.25)
    plt.tight_layout()
    _save(fig, save_path)


def plot_method_radar(raw_df, model_name: str, save_path: str) -> None:
    """
    Radar (spider) chart comparing XAI methods across four normalised
    cost dimensions: latency, RAM, VRAM, and CV stability.
    """
    methods = list(raw_df["Method"].unique())
    stats = {}
    for m in methods:
        sub  = raw_df[raw_df["Method"] == m]
        lat  = sub["Latency (s)"].values
        ram  = sub["Peak RAM (MB)"].values
        vram = sub["Peak VRAM (MB)"].values
        cv   = lat.std() / lat.mean() if lat.mean() > 0 else 0.0
        stats[m] = {"Latency": lat.mean(), "Peak RAM": ram.mean(),
                    "Peak VRAM": vram.mean(), "CV (stability)": cv}

    categories = ["Latency", "Peak RAM", "Peak VRAM", "CV (stability)"]
    N       = len(categories)
    cat_max = {c: max(stats[m][c] for m in methods) or 1.0 for c in categories}
    angles  = [n / float(N) * 2 * np.pi for n in range(N)] +               [0 / float(N) * 2 * np.pi]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, m in enumerate(methods):
        color  = _method_color(m, i)
        values = [stats[m][c] / cat_max[c] for c in categories]
        values += [values[0]]
        ax.plot(angles, values, "o-", linewidth=2, color=color,
                label=m, markersize=5)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.grid(color="grey", ls="--", alpha=0.4)
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — XAI Method Radar\n"
        "(normalised cost — outer = higher cost)",
        fontsize=12, fontweight="bold", pad=18,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=10, frameon=False)
    plt.tight_layout()
    _save(fig, save_path)


def plot_cumulative_time(raw_df, model_name: str, save_path: str) -> None:
    """
    Cumulative explanation time over sequential runs.
    Shows the total time-budget required for N explanations.
    """
    methods = list(raw_df["Method"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        color = _method_color(m, i)
        sub   = raw_df[raw_df["Method"] == m].sort_values("Run")
        cum   = np.cumsum(sub["Latency (s)"].values)
        runs  = sub["Run"].values
        ax.plot(runs, cum, "o-", color=color, linewidth=2.2,
                markersize=5, label=m, alpha=0.88)
        ax.annotate(f"{cum[-1]:.2f}s", (runs[-1], cum[-1]),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=9, fontweight="bold", color=color)
    ax.set_xlabel("Run Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Explanation Time (s)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} — Cumulative Time Budget\n"
        "Total wall-clock cost of N sequential XAI explanations",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, save_path)


def plot_cross_model_comparison(
    report_paths: dict,
    metric: str = "Latency Mean (s)",
    save_path: str = "phase_5_evaluation/results/cross_model_latency_comparison.png",
) -> None:
    """
    Grouped bar chart comparing an aggregated metric across multiple models.
    Essential for the model-comparison section — which backbone has lower XAI overhead?

    Parameters
    ----------
    report_paths : Dict  model_name -> path to *_benchmark_report.csv
    metric       : Column name in the CSV to compare.
    save_path    : Output PNG path.
    """
    import pandas as pd

    model_dfs = {}
    for model_name, path in report_paths.items():
        if os.path.exists(path):
            model_dfs[model_name] = pd.read_csv(path)
        else:
            print(f"  [skip] {path} not found — run that model first.")

    if len(model_dfs) < 2:
        print("  [skip] Need >= 2 model reports for cross-model comparison.")
        return

    all_methods = list(next(iter(model_dfs.values()))["Method"].tolist())
    n_models    = len(model_dfs)
    x = np.arange(len(all_methods))
    w = 0.8 / n_models
    model_colors = ["#3A86FF", "#FF006E", "#8338EC", "#FFBE0B"]

    fig, ax = plt.subplots(figsize=(max(8, len(all_methods) * 1.4), 5.5))
    for j, (model_name, df) in enumerate(model_dfs.items()):
        vals = []
        for m in all_methods:
            row = df[df["Method"] == m]
            vals.append(float(row[metric].values[0]) if len(row) else 0.0)
        color  = model_colors[j % len(model_colors)]
        offset = (j - n_models / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w,
                      label=model_name.replace("_", " ").title(),
                      color=color, edgecolor="white", linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.4f}" if v < 1 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, fontsize=11)
    ax.set_ylabel(metric, fontsize=11, fontweight="bold")
    ax.set_title(
        f"Cross-Model Comparison — {metric}\n"
        "MobileNetV3 vs ShuffleNetV2 per XAI method",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
