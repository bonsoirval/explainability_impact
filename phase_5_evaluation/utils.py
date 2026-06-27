import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Check if there is VRAM usage
    has_vram = (raw_df['Peak VRAM (MB)'].sum() > 0)
    num_plots = 3 if has_vram else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5.5))
    if num_plots == 1:
        axes = [axes]
        
    # Styling settings
    colors = ['#3A86FF', '#FF006E', '#8338EC', '#FFBE0B']
    box_props = dict(patch_artist=True, medianprops=dict(color='black', linewidth=1.5), flierprops=dict(marker='o', markersize=4, alpha=0.5))
    
    # Plot Latency (Log scale)
    ax_lat = axes[0]
    data_lat = [raw_df[raw_df['Method'] == m]['Latency (s)'].values for m in methods]
    bp_lat = ax_lat.boxplot(data_lat, tick_labels=list(methods), **box_props)
    
    # Color boxes and add jitter
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
    
    # Plot Peak RAM
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
    
    # Plot Peak VRAM if needed
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
        
    # Formatting adjustments
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
