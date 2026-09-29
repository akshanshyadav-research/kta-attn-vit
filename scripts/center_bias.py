"""Notebook-derived analysis; run only as a command-line script."""
from kta_attn.analysis_setup import setup
globals().update(setup())
import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
MAX_IMAGES = args.max_images
KTA_KEEP_RATIO = drop_rate
GRID_SIZE = 14
NUM_PATCHES = GRID_SIZE * GRID_SIZE
NUM_RANDOM_TRIALS = 5
CENTER_THRESHOLD = 0.25
OUTPUT_DIR = str(args.output)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('=' * 70)
print('STAGE-0 KTA CENTER-BIAS ANALYSIS')
print('=' * 70)
print('Device:', device)
print('ImageNet folder:', img_folder)
print('KTA keep ratio:', KTA_KEEP_RATIO)
print('Number of patches:', NUM_PATCHES)
print('Grid:', f'{GRID_SIZE} x {GRID_SIZE}')
print('Maximum images:', MAX_IMAGES)
print()
(yy, xx) = torch.meshgrid(torch.arange(GRID_SIZE), torch.arange(GRID_SIZE), indexing='ij')
xx = xx.float()
yy = yy.float()
center_x = (GRID_SIZE - 1) / 2.0
center_y = (GRID_SIZE - 1) / 2.0
center_distance = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
center_distance = center_distance / center_distance.max()
center_distance_flat = center_distance.flatten()
print('Center-distance range:')
print('min =', center_distance_flat.min().item(), 'max =', center_distance_flat.max().item())
print()
captured_stage0 = {}

def stage0_pre_hook(module, inputs):
    """
    Capture the exact tensor entering your PatchDropout module.
    """
    captured_stage0['x'] = inputs[0].detach()
patch_drop_handle = quantized_model.model.patch_drop.register_forward_pre_hook(stage0_pre_hook)

def get_exact_kta_selection(stage0_x):
    """
    Reproduce the KTA selection used in your notebook.

    stage0_x:
        [1, 197, embedding_dim]

    Returns:
        selected_indices:
            indices of selected patch tokens [0 ... 195]

        kta_scores:
            Kendall-tau score for every patch
    """
    cls_token = stage0_x[:, 0, :]
    patch_tokens = stage0_x[:, 1:, :]
    distances = pearson_corr(cls_token, patch_tokens)
    seq_len = stage0_x.shape[1]
    k = int((seq_len - 1) * KTA_KEEP_RATIO)
    (threshold_values, _) = torch.kthvalue(distances, min(k + 1, distances.shape[1]), dim=1, keepdim=True)
    keep_mask = distances <= threshold_values
    selected_indices = torch.nonzero(keep_mask[0], as_tuple=False).flatten()
    return (selected_indices.cpu(), distances[0].detach().cpu())

def get_attention_selection(stage0_x, model, num_keep):
    """
    Compute Block-0 CLS attention and select the same number
    of patch tokens as KTA.
    """
    block0 = model.model.blocks[0]
    x_norm = block0.norm1(stage0_x)
    (B, N, C) = x_norm.shape
    qkv = block0.attn.qkv(x_norm).reshape(B, N, 3, block0.attn.num_heads, C // block0.attn.num_heads).permute(2, 0, 3, 1, 4)
    q = qkv[0]
    k = qkv[1]
    attn = q @ k.transpose(-2, -1) * block0.attn.scale
    attn = torch.softmax(attn, dim=-1)
    avg_attn = attn.mean(dim=1)
    cls_attention = avg_attn[:, 0, :]
    patch_attention = cls_attention[:, 1:]
    selected_indices = torch.topk(patch_attention[0], k=num_keep, largest=True).indices
    return (selected_indices.cpu(), patch_attention[0].detach().cpu())

def get_random_selection(num_keep):
    """
    Randomly select exactly the same number of patches as KTA.
    """
    return torch.randperm(NUM_PATCHES)[:num_keep]

def calculate_selection_statistics(selected_indices):
    """
    Calculate spatial statistics for selected patches.
    """
    distances = center_distance_flat[selected_indices]
    return {'mean_center_distance': distances.mean().item(), 'median_center_distance': distances.median().item(), 'std_center_distance': distances.std().item(), 'min_center_distance': distances.min().item(), 'max_center_distance': distances.max().item(), 'center_patch_fraction': (distances <= CENTER_THRESHOLD).float().mean().item(), 'num_selected': len(selected_indices)}
image_files = []
print('Collecting ImageNet validation images...')
for class_name in sorted(os.listdir(img_folder)):
    class_path = os.path.join(img_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    for img_name in sorted(os.listdir(class_path)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(class_path, img_name)
        image_files.append((img_path, class_name, img_name))
        if MAX_IMAGES is not None and len(image_files) >= MAX_IMAGES:
            break
    if MAX_IMAGES is not None and len(image_files) >= MAX_IMAGES:
        break
print('Images selected for analysis:', len(image_files))
print()
results = []
all_kta_distances = []
all_random_distances = []
all_attention_distances = []
kta_heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
random_heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
attention_heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
quantized_model.eval()
for (image_number, (img_path, class_name, img_name)) in enumerate(image_files, start=1):
    image = cv2.imread(img_path)
    if image is None:
        print('Could not load:', img_path)
        continue
    (image_tensor, fast_mask) = main_processing(image)
    captured_stage0.clear()
    with torch.no_grad():
        _ = quantized_model(image_tensor, mask=fast_mask)
    stage0_x = captured_stage0['x']
    assert stage0_x.shape[1] == 197, f'Expected 197 tokens, got {stage0_x.shape}'
    (kta_indices, kta_scores) = get_exact_kta_selection(stage0_x)
    num_keep = len(kta_indices)
    random_indices_trials = []
    for _ in range(NUM_RANDOM_TRIALS):
        random_indices = get_random_selection(num_keep)
        random_indices_trials.append(random_indices)
    random_indices = random_indices_trials[0]
    (attention_indices, attention_scores) = get_attention_selection(stage0_x, quantized_model, num_keep)
    kta_stats = calculate_selection_statistics(kta_indices)
    random_stats_trials = []
    for random_indices_trial in random_indices_trials:
        random_stats_trials.append(calculate_selection_statistics(random_indices_trial))
    random_stats = {}
    for key in random_stats_trials[0].keys():
        if key == 'num_selected':
            random_stats[key] = random_stats_trials[0][key]
        else:
            random_stats[key] = np.mean([x[key] for x in random_stats_trials])
    attention_stats = calculate_selection_statistics(attention_indices)
    all_kta_distances.extend(center_distance_flat[kta_indices].numpy())
    all_random_distances.extend(center_distance_flat[random_indices].numpy())
    all_attention_distances.extend(center_distance_flat[attention_indices].numpy())
    for idx in kta_indices.numpy():
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        kta_heatmap[row, col] += 1
    for idx in random_indices.numpy():
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        random_heatmap[row, col] += 1
    for idx in attention_indices.numpy():
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        attention_heatmap[row, col] += 1
    result = {'image': img_name, 'class': class_name, 'image_path': img_path, 'num_kta_selected': num_keep, 'kta_mean_center_distance': kta_stats['mean_center_distance'], 'kta_median_center_distance': kta_stats['median_center_distance'], 'kta_std_center_distance': kta_stats['std_center_distance'], 'kta_center_patch_fraction': kta_stats['center_patch_fraction'], 'random_mean_center_distance': random_stats['mean_center_distance'], 'random_median_center_distance': random_stats['median_center_distance'], 'random_center_patch_fraction': random_stats['center_patch_fraction'], 'attention_mean_center_distance': attention_stats['mean_center_distance'], 'attention_median_center_distance': attention_stats['median_center_distance'], 'attention_center_patch_fraction': attention_stats['center_patch_fraction']}
    results.append(result)
    if image_number % 10 == 0:
        print(f'Processed {image_number}/{len(image_files)} images')
patch_drop_handle.remove()
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, 'imagenet_kta_center_bias_results.csv')
results_df.to_csv(csv_path, index=False)
print()
print('Per-image results saved to:', csv_path)
kta_distances = np.array(all_kta_distances)
random_distances = np.array(all_random_distances)
attention_distances = np.array(all_attention_distances)
print()
print('=' * 70)
print('OVERALL CENTER-BIAS RESULTS')
print('=' * 70)
print()
print('Number of images:', len(results_df))
print()
print('Mean normalized center distance:')
print(f'  All patches: {center_distance_flat.mean().item():.6f}')
print(f'  KTA: {kta_distances.mean():.6f}')
print(f'  Random: {random_distances.mean():.6f}')
print(f'  Attention: {attention_distances.mean():.6f}')
print()
print('Median normalized center distance:')
print(f'  KTA: {np.median(kta_distances):.6f}')
print(f'  Random: {np.median(random_distances):.6f}')
print(f'  Attention: {np.median(attention_distances):.6f}')
kta_mean = kta_distances.mean()
random_mean = random_distances.mean()
attention_mean = attention_distances.mean()
print()
print('Difference from random:')
print(f'  KTA - Random: {kta_mean - random_mean:.6f}')
print(f'  Attention - Random: {attention_mean - random_mean:.6f}')
print()
print('Relative center-distance ratio:')
print(f'  KTA / Random: {kta_mean / random_mean:.6f}')
print(f'  Attention / Random: {attention_mean / random_mean:.6f}')
kta_center_fraction = np.mean(kta_distances <= CENTER_THRESHOLD)
random_center_fraction = np.mean(random_distances <= CENTER_THRESHOLD)
attention_center_fraction = np.mean(attention_distances <= CENTER_THRESHOLD)
all_center_fraction = np.mean(center_distance_flat.numpy() <= CENTER_THRESHOLD)
print()
print('=' * 70)
print('CENTER REGION ANALYSIS')
print('=' * 70)
print(f'Center threshold: {CENTER_THRESHOLD}')
print()
print(f'All patches: {all_center_fraction:.6f}')
print(f'KTA: {kta_center_fraction:.6f}')
print(f'Random: {random_center_fraction:.6f}')
print(f'Attention: {attention_center_fraction:.6f}')
print()
print('Relative center-selection preference:')
print(f'  KTA / Random = {kta_center_fraction / random_center_fraction:.6f}x')
print(f'  Attention / Random = {attention_center_fraction / random_center_fraction:.6f}x')
summary = {'num_images': len(results_df), 'num_patches': NUM_PATCHES, 'kta_keep_ratio': KTA_KEEP_RATIO, 'kta_mean_center_distance': float(kta_mean), 'random_mean_center_distance': float(random_mean), 'attention_mean_center_distance': float(attention_mean), 'kta_median_center_distance': float(np.median(kta_distances)), 'random_median_center_distance': float(np.median(random_distances)), 'attention_median_center_distance': float(np.median(attention_distances)), 'kta_random_distance_ratio': float(kta_mean / random_mean), 'attention_random_distance_ratio': float(attention_mean / random_mean), 'kta_center_fraction': float(kta_center_fraction), 'random_center_fraction': float(random_center_fraction), 'attention_center_fraction': float(attention_center_fraction), 'kta_random_center_selection_ratio': float(kta_center_fraction / random_center_fraction), 'center_threshold': CENTER_THRESHOLD}
summary_df = pd.DataFrame([summary])
summary_path = os.path.join(OUTPUT_DIR, 'imagenet_kta_center_bias_summary.csv')
summary_df.to_csv(summary_path, index=False)
plt.figure(figsize=(9, 6))
plt.hist(random_distances, bins=40, density=True, alpha=0.45, label='Random')
plt.hist(kta_distances, bins=40, density=True, alpha=0.55, label='KTA')
plt.hist(attention_distances, bins=40, density=True, alpha=0.45, label='Attention')
plt.axvline(kta_mean, linestyle='--', label=f'KTA mean = {kta_mean:.3f}')
plt.axvline(random_mean, linestyle='--', label=f'Random mean = {random_mean:.3f}')
plt.xlabel('Normalized distance from image center')
plt.ylabel('Density')
plt.title('Spatial Distribution of Selected Patches')
plt.legend()
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, 'center_distance_distribution.png')
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.show()

def plot_heatmap(heatmap, title, filename):
    heatmap_normalized = heatmap / len(results_df)
    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap_normalized, interpolation='nearest')
    plt.colorbar(label='Selection frequency')
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))
    plt.xlabel('Patch column')
    plt.ylabel('Patch row')
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
plot_heatmap(kta_heatmap, 'Spatial Distribution of KTA-Selected Patches', 'kta_spatial_heatmap.png')
plot_heatmap(random_heatmap, 'Spatial Distribution of Randomly Selected Patches', 'random_spatial_heatmap.png')
plot_heatmap(attention_heatmap, 'Spatial Distribution of Attention-Selected Patches', 'attention_spatial_heatmap.png')
print()
print('=' * 70)
print('INTERPRETATION')
print('=' * 70)
if kta_mean < random_mean:
    print('KTA selects patches that are, on average, closer to the image center than random selection.')
    print('This indicates some degree of center preference.')
else:
    print('KTA does not show a stronger center preference than random selection based on mean center distance.')
if kta_center_fraction > random_center_fraction:
    print('KTA selects a larger fraction of patches from the central region than random selection.')
else:
    print('KTA does not select a larger fraction of central patches than random selection.')
print()
print('KTA/Random center-distance ratio:', f'{kta_mean / random_mean:.4f}x')
print('KTA/Random center-selection ratio:', f'{kta_center_fraction / random_center_fraction:.4f}x')
print()
print('Results directory:', OUTPUT_DIR)
print('Summary:', summary_path)
print('Detailed CSV:', csv_path)
print('=' * 70)
