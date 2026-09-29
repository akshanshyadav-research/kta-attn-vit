"""Notebook-derived analysis; run only as a command-line script."""
from kta_attn.analysis_setup import setup
globals().update(setup())
import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
MAX_IMAGES = args.max_images
TRANSLATIONS = [-48, -32, -16, 0, +16, +32, +48]
IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE
CENTER_THRESHOLD = 0.25
OUTPUT_DIR = str(args.output)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('=' * 80)
print('KTA CENTER-BIAS VS TRANSLATION-CONSISTENCY ANALYSIS')
print('=' * 80)
print('ImageNet folder:', img_folder)
print('Device:', device)
print('Patch grid:', f'{GRID_SIZE} x {GRID_SIZE}')
print('Number of patches:', NUM_PATCHES)
print('Patch size:', PATCH_SIZE)
print('KTA keep ratio:', drop_rate)
print('Translations:', TRANSLATIONS)
print('Maximum images:', MAX_IMAGES)
print()
(patch_rows, patch_cols) = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
center_x = (GRID_SIZE - 1) / 2.0
center_y = (GRID_SIZE - 1) / 2.0
center_distance_grid = np.sqrt((patch_cols - center_x) ** 2 + (patch_rows - center_y) ** 2)
center_distance_grid = center_distance_grid / center_distance_grid.max()
center_distance = center_distance_grid.flatten()
print('Center distance range:', center_distance.min(), 'to', center_distance.max())
print()

def preprocess_image(image):
    image = resize_nearest_neighbor(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = bgr_to_rgb(image)
    image = image.copy()
    tensor = transform_image(image, device)
    return tensor

def translate_image(image, shift_x):
    (height, width) = image.shape[:2]
    translated = np.zeros_like(image)
    if shift_x == 0:
        return image.copy()
    if shift_x > 0:
        translated[:, shift_x:] = image[:, :width - shift_x]
    else:
        shift = abs(shift_x)
        translated[:, :width - shift] = image[:, shift:]
    return translated
captured_stage0 = {}

def capture_stage0(module, inputs):
    captured_stage0['x'] = inputs[0].detach()
hook_handle = quantized_model.model.patch_drop.register_forward_pre_hook(capture_stage0)

def exact_kta_selection(stage0_x):
    cls_token = stage0_x[:, 0, :]
    patch_tokens = stage0_x[:, 1:, :]
    assert patch_tokens.shape[1] == NUM_PATCHES, f'Unexpected number of patches: {patch_tokens.shape[1]}'
    distances = pearson_corr(cls_token, patch_tokens)
    seq_len = stage0_x.shape[1]
    k = int((seq_len - 1) * drop_rate)
    (threshold_values, _) = torch.kthvalue(distances, min(k + 1, distances.shape[1]), dim=1, keepdim=True)
    keep_mask = distances <= threshold_values
    mask = keep_mask[0].detach().cpu().numpy().astype(bool)
    selected_indices = np.where(mask)[0]
    return (selected_indices, distances[0].detach().cpu().numpy())
print('=' * 80)
print('VERIFYING KTA')
print('=' * 80)
first_image_path = None
for class_name in sorted(os.listdir(img_folder)):
    class_path = os.path.join(img_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    for image_name in sorted(os.listdir(class_path)):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            first_image_path = os.path.join(class_path, image_name)
            break
    if first_image_path is not None:
        break
if first_image_path is None:
    hook_handle.remove()
    raise RuntimeError('No ImageNet images found.')
first_image = cv2.imread(first_image_path)
first_tensor = preprocess_image(first_image)
first_mask = np.random.randint(1, 2, size=197)
captured_stage0.clear()
with torch.no_grad():
    _ = quantized_model(first_tensor, mask=first_mask)
first_stage0 = captured_stage0['x']
print('Stage-0 tensor:', first_stage0.shape)
(first_indices, _) = exact_kta_selection(first_stage0)
print('KTA selected:', len(first_indices))
print('Expected:', int(NUM_PATCHES * drop_rate))
print()

def translate_patch_indices(indices, shift_x):
    patch_shift = shift_x // PATCH_SIZE
    translated_indices = []
    for index in indices:
        row = index // GRID_SIZE
        col = index % GRID_SIZE
        new_col = col + patch_shift
        if 0 <= new_col < GRID_SIZE:
            new_index = row * GRID_SIZE + new_col
            translated_indices.append(new_index)
    return np.array(translated_indices, dtype=np.int64)

def calculate_overlap(original_indices, shifted_indices, shift_x):
    expected_indices = translate_patch_indices(original_indices, shift_x)
    A = set(expected_indices.tolist())
    B = set(shifted_indices.tolist())
    intersection = A.intersection(B)
    union = A.union(B)
    if len(union) > 0:
        iou = len(intersection) / len(union)
    else:
        iou = 0.0
    if len(B) > 0:
        precision = len(intersection) / len(B)
    else:
        precision = 0.0
    if len(A) > 0:
        recall = len(intersection) / len(A)
    else:
        recall = 0.0
    if len(A) + len(B) > 0:
        dice = 2.0 * len(intersection) / (len(A) + len(B))
    else:
        dice = 0.0
    return {'expected_count': len(A), 'actual_count': len(B), 'intersection': len(intersection), 'union': len(union), 'iou': iou, 'precision': precision, 'recall': recall, 'dice': dice}

def spatial_statistics(selected_indices):
    distances = center_distance[selected_indices]
    return {'mean_center_distance': float(np.mean(distances)), 'median_center_distance': float(np.median(distances)), 'center_fraction': float(np.mean(distances <= CENTER_THRESHOLD)), 'min_center_distance': float(np.min(distances)), 'max_center_distance': float(np.max(distances))}
image_files = []
for class_name in sorted(os.listdir(img_folder)):
    class_path = os.path.join(img_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    for image_name in sorted(os.listdir(class_path)):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_files.append((os.path.join(class_path, image_name), class_name, image_name))
        if MAX_IMAGES is not None and len(image_files) >= MAX_IMAGES:
            break
    if MAX_IMAGES is not None and len(image_files) >= MAX_IMAGES:
        break
print('Images selected:', len(image_files))
print()
results = []
quantized_model.eval()
for (image_number, (image_path, class_name, image_name)) in enumerate(image_files, start=1):
    image = cv2.imread(image_path)
    if image is None:
        continue
    image = resize_nearest_neighbor(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = bgr_to_rgb(image)
    image = image.copy()
    fast_mask = np.random.randint(1, 2, size=197)
    selections = {}
    for shift_x in TRANSLATIONS:
        translated = translate_image(image, shift_x)
        tensor = transform_image(translated, device)
        captured_stage0.clear()
        with torch.no_grad():
            _ = quantized_model(tensor, mask=fast_mask)
        stage0_x = captured_stage0['x']
        (selected_indices, kta_scores) = exact_kta_selection(stage0_x)
        selections[shift_x] = selected_indices
    original_indices = selections[0]
    original_stats = spatial_statistics(original_indices)
    random_indices = np.random.choice(NUM_PATCHES, size=len(original_indices), replace=False)
    random_stats = spatial_statistics(random_indices)
    results.append({'image': image_name, 'class': class_name, 'translation_pixels': 0, 'patch_shift': 0, 'num_selected': len(original_indices), 'kta_mean_center_distance': original_stats['mean_center_distance'], 'kta_median_center_distance': original_stats['median_center_distance'], 'kta_center_fraction': original_stats['center_fraction'], 'random_mean_center_distance': random_stats['mean_center_distance'], 'random_center_fraction': random_stats['center_fraction'], 'translation_iou': 1.0, 'translation_precision': 1.0, 'translation_recall': 1.0, 'translation_dice': 1.0})
    for shift_x in TRANSLATIONS:
        if shift_x == 0:
            continue
        shifted_indices = selections[shift_x]
        shifted_stats = spatial_statistics(shifted_indices)
        overlap = calculate_overlap(original_indices, shifted_indices, shift_x)
        results.append({'image': image_name, 'class': class_name, 'translation_pixels': shift_x, 'patch_shift': shift_x // PATCH_SIZE, 'num_selected': len(shifted_indices), 'kta_mean_center_distance': shifted_stats['mean_center_distance'], 'kta_median_center_distance': shifted_stats['median_center_distance'], 'kta_center_fraction': shifted_stats['center_fraction'], 'random_mean_center_distance': random_stats['mean_center_distance'], 'random_center_fraction': random_stats['center_fraction'], 'translation_iou': overlap['iou'], 'translation_precision': overlap['precision'], 'translation_recall': overlap['recall'], 'translation_dice': overlap['dice']})
    if image_number % 10 == 0:
        print(f'Processed {image_number}/{len(image_files)}')
hook_handle.remove()
results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, 'kta_translation_per_image.csv')
results_df.to_csv(results_path, index=False)
summary = results_df.groupby(['translation_pixels', 'patch_shift']).agg(num_images=('image', 'count'), mean_center_distance=('kta_mean_center_distance', 'mean'), std_center_distance=('kta_mean_center_distance', 'std'), median_center_distance=('kta_median_center_distance', 'mean'), center_patch_fraction=('kta_center_fraction', 'mean'), random_center_distance=('random_mean_center_distance', 'mean'), random_center_fraction=('random_center_fraction', 'mean'), mean_translation_iou=('translation_iou', 'mean'), std_translation_iou=('translation_iou', 'std'), mean_precision=('translation_precision', 'mean'), mean_recall=('translation_recall', 'mean'), mean_dice=('translation_dice', 'mean')).reset_index()
summary_path = os.path.join(OUTPUT_DIR, 'kta_translation_summary.csv')
summary.to_csv(summary_path, index=False)
print()
print('=' * 80)
print('FINAL TRANSLATION RESULTS')
print('=' * 80)
print()
print(summary.to_string(index=False))
original = summary[summary['translation_pixels'] == 0]
if len(original) > 0:
    original_center_distance = original['mean_center_distance'].iloc[0]
    random_center_distance = original['random_center_distance'].iloc[0]
    original_center_fraction = original['center_patch_fraction'].iloc[0]
    random_center_fraction = original['random_center_fraction'].iloc[0]
    print()
    print('=' * 80)
    print('CENTER-BIAS RESULT ON NATURAL IMAGENET IMAGES')
    print('=' * 80)
    print()
    print(f'KTA center distance: {original_center_distance:.6f}')
    print(f'Random center distance: {random_center_distance:.6f}')
    print(f'KTA / Random distance ratio: {original_center_distance / random_center_distance:.6f}')
    print()
    print(f'KTA center fraction: {original_center_fraction:.6f}')
    print(f'Random center fraction: {random_center_fraction:.6f}')
    print(f'KTA / Random center fraction ratio: {original_center_fraction / random_center_fraction:.6f}')
nonzero = summary[summary['translation_pixels'] != 0]
print()
print('=' * 80)
print('TRANSLATION-COMPENSATED TOKEN CONSISTENCY')
print('=' * 80)
print()
for (_, row) in nonzero.iterrows():
    print(f"{int(row['translation_pixels']):+d} px ({int(row['patch_shift']):+d} patches): IoU = {row['mean_translation_iou']:.4f}, Precision = {row['mean_precision']:.4f}, Recall = {row['mean_recall']:.4f}, Dice = {row['mean_dice']:.4f}")
plt.figure(figsize=(9, 6))
plt.plot(summary['translation_pixels'], summary['mean_center_distance'], marker='o', linewidth=2, label='KTA')
plt.plot(summary['translation_pixels'], summary['random_center_distance'], marker='s', linewidth=2, label='Random')
plt.axhline(center_distance.mean(), linestyle='--', label='All patches')
plt.xlabel('Horizontal translation (pixels)')
plt.ylabel('Mean normalized distance from image center')
plt.title('KTA Center Preference Under Spatial Translation')
plt.xticks(TRANSLATIONS)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
center_plot_path = os.path.join(OUTPUT_DIR, 'center_distance_vs_translation.png')
plt.savefig(center_plot_path, dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(9, 6))
plt.plot(summary['translation_pixels'], summary['center_patch_fraction'], marker='o', linewidth=2, label='KTA')
plt.plot(summary['translation_pixels'], summary['random_center_fraction'], marker='s', linewidth=2, label='Random')
plt.xlabel('Horizontal translation (pixels)')
plt.ylabel('Fraction of selected patches in center region')
plt.title('Central Patch Selection Under Spatial Translation')
plt.xticks(TRANSLATIONS)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
center_fraction_path = os.path.join(OUTPUT_DIR, 'center_fraction_vs_translation.png')
plt.savefig(center_fraction_path, dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(9, 6))
plt.plot(nonzero['translation_pixels'], nonzero['mean_translation_iou'], marker='o', linewidth=2, label='IoU')
plt.plot(nonzero['translation_pixels'], nonzero['mean_dice'], marker='s', linewidth=2, label='Dice')
plt.xlabel('Horizontal translation (pixels)')
plt.ylabel('Selection consistency')
plt.title('KTA Translation-Consistent Token Selection')
plt.xticks(TRANSLATIONS)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
iou_plot_path = os.path.join(OUTPUT_DIR, 'translation_consistency.png')
plt.savefig(iou_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print()
print('=' * 80)
print('INTERPRETATION')
print('=' * 80)
print()
if len(original) > 0:
    ratio = original_center_distance / random_center_distance
    center_ratio = original_center_fraction / random_center_fraction
    print(f"On natural ImageNet images, KTA's mean center distance is {ratio:.3f}x that of random selection.")
    print(f'KTA selects center-region patches at {center_ratio:.3f}x the frequency of random selection.')
print()
if len(nonzero) > 0:
    mean_iou = nonzero['mean_translation_iou'].mean()
    mean_recall = nonzero['mean_recall'].mean()
    print(f'Mean translation-compensated IoU across non-zero translations: {mean_iou:.4f}')
    print(f'Mean translation-compensated recall: {mean_recall:.4f}')
print()
print('Interpretation:')
print()
print('A center preference alone does not prove that KTA uses a fixed center prior.')
print('High translation-compensated IoU/recall indicates that KTA-selected tokens move consistently with the translated visual content.')
print()
print('Therefore, if the center preference remains present but translation-compensated overlap is high, the evidence supports the conclusion that KTA has a moderate spatial preference on natural ImageNet images but is not solely determined by an absolute image-center prior.')
print()
print('Output directory:', OUTPUT_DIR)
print('Per-image results:', results_path)
print('Summary:', summary_path)
print('Center-distance plot:', center_plot_path)
print('Center-fraction plot:', center_fraction_path)
print('Translation-consistency plot:', iou_plot_path)
print('=' * 80)
