import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

