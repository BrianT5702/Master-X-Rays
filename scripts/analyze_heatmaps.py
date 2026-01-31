"""Analyze Grad-CAM heatmaps for accuracy assessment."""

from __future__ import annotations

import sys
import io
from pathlib import Path
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_filename(filename: str) -> dict:
    """Parse heatmap filename to extract metadata."""
    # Format: sample_XXXX_class_NAME_pred_X.XXX_gt_X.png
    match = re.match(r'sample_(\d+)_class_(\w+)_pred_([\d.]+)_gt_(\d+)\.png', filename)
    if match:
        return {
            'sample_idx': int(match.group(1)),
            'class_name': match.group(2),
            'pred_prob': float(match.group(3)),
            'gt_label': int(match.group(4)),
        }
    return None


def analyze_heatmap_region(heatmap_path: Path) -> dict:
    """Analyze where the heatmap focuses (center vs edges)."""
    img = Image.open(heatmap_path)
    img_array = np.array(img.convert('RGB'))
    
    h, w = img_array.shape[:2]
    center_h, center_w = h // 2, w // 2
    
    # Extract the heatmap (middle panel) - it's the second of three panels
    # Each panel is approximately w/3 wide
    panel_width = w // 3
    heatmap_start = panel_width
    heatmap_end = panel_width * 2
    
    # Extract heatmap region (middle panel)
    heatmap_region = img_array[:, heatmap_start:heatmap_end]
    
    # Convert to grayscale for intensity analysis
    if len(heatmap_region.shape) == 3:
        heatmap_gray = np.mean(heatmap_region, axis=2)
    else:
        heatmap_gray = heatmap_region
    
    # Normalize
    heatmap_gray = heatmap_gray.astype(np.float32) / 255.0
    
    # Analyze activation distribution
    # High values indicate where the model focuses
    threshold = 0.5  # 50% intensity threshold
    
    # Center region (middle 60% of image)
    center_margin_h = int(h * 0.2)
    center_margin_w = int(heatmap_region.shape[1] * 0.2)
    center_region = heatmap_gray[center_margin_h:h-center_margin_h, 
                                 center_margin_w:heatmap_region.shape[1]-center_margin_w]
    
    # Edge regions (outer 20% on each side)
    top_edge = heatmap_gray[:center_margin_h, :]
    bottom_edge = heatmap_gray[h-center_margin_h:, :]
    left_edge = heatmap_gray[:, :center_margin_w]
    right_edge = heatmap_gray[:, heatmap_region.shape[1]-center_margin_w:]
    
    center_intensity = np.mean(center_region)
    edge_intensity = np.mean([np.mean(top_edge), np.mean(bottom_edge), 
                              np.mean(left_edge), np.mean(right_edge)])
    
    # Calculate how much activation is in center vs edges
    center_ratio = center_intensity / (center_intensity + edge_intensity + 1e-8)
    
    # Check if heatmap is focused (high center ratio = good, low = bad/artifacts)
    is_focused = center_ratio > 0.6
    
    return {
        'center_intensity': center_intensity,
        'edge_intensity': edge_intensity,
        'center_ratio': center_ratio,
        'is_focused': is_focused,
        'max_intensity': np.max(heatmap_gray),
        'mean_intensity': np.mean(heatmap_gray),
    }


def main():
    heatmap_dir = Path("heatmaps/test/gradcam")
    
    if not heatmap_dir.exists():
        print(f"ERROR: Heatmap directory not found: {heatmap_dir}")
        return
    
    heatmap_files = sorted(heatmap_dir.glob("*.png"))
    
    if not heatmap_files:
        print(f"ERROR: No heatmap files found in {heatmap_dir}")
        return
    
    print(f"Analyzing {len(heatmap_files)} heatmap files...\n")
    
    # Statistics
    stats = {
        'true_positives': [],
        'false_positives': [],
        'true_negatives': [],
        'false_negatives': [],
        'by_class': defaultdict(list),
    }
    
    focused_count = 0
    unfocused_count = 0
    
    for heatmap_path in heatmap_files:
        metadata = parse_filename(heatmap_path.name)
        if not metadata:
            continue
        
        analysis = analyze_heatmap_region(heatmap_path)
        
        # Classify prediction
        pred_positive = metadata['pred_prob'] > 0.3  # Using threshold from config
        gt_positive = metadata['gt_label'] == 1
        
        if pred_positive and gt_positive:
            stats['true_positives'].append((metadata, analysis))
        elif pred_positive and not gt_positive:
            stats['false_positives'].append((metadata, analysis))
        elif not pred_positive and not gt_positive:
            stats['true_negatives'].append((metadata, analysis))
        else:
            stats['false_negatives'].append((metadata, analysis))
        
        stats['by_class'][metadata['class_name']].append((metadata, analysis))
        
        if analysis['is_focused']:
            focused_count += 1
        else:
            unfocused_count += 1
    
    # Print summary
    print("=" * 60)
    print("HEATMAP ACCURACY ANALYSIS")
    print("=" * 60)
    
    print(f"\nPrediction Statistics:")
    print(f"   True Positives:  {len(stats['true_positives'])}")
    print(f"   False Positives: {len(stats['false_positives'])}")
    print(f"   True Negatives:  {len(stats['true_negatives'])}")
    print(f"   False Negatives: {len(stats['false_negatives'])}")
    
    print(f"\nHeatmap Focus Analysis:")
    print(f"   Focused on center (lung fields): {focused_count} ({focused_count/len(heatmap_files)*100:.1f}%)")
    print(f"   Unfocused (artifacts/edges):     {unfocused_count} ({unfocused_count/len(heatmap_files)*100:.1f}%)")
    
    # Analyze by class
    print(f"\nBy Class:")
    for class_name in ['Nodule', 'Fibrosis']:
        if class_name in stats['by_class']:
            class_data = stats['by_class'][class_name]
            focused = sum(1 for _, a in class_data if a['is_focused'])
            avg_center_ratio = np.mean([a['center_ratio'] for _, a in class_data])
            print(f"   {class_name}:")
            print(f"      Total: {len(class_data)}")
            print(f"      Focused: {focused} ({focused/len(class_data)*100:.1f}%)")
            print(f"      Avg center ratio: {avg_center_ratio:.3f}")
    
    # Analyze false positives (most concerning)
    if stats['false_positives']:
        print(f"\nWARNING: FALSE POSITIVES Analysis (gt=0, pred>0.3):")
        fp_focused = sum(1 for _, a in stats['false_positives'] if a['is_focused'])
        fp_unfocused = len(stats['false_positives']) - fp_focused
        print(f"   Total FPs: {len(stats['false_positives'])}")
        print(f"   Focused on center: {fp_focused}")
        print(f"   Unfocused (likely artifacts): {fp_unfocused}")
        
        if fp_unfocused > 0:
            print(f"\n   WARNING: {fp_unfocused} false positives have heatmaps focused on edges/artifacts!")
            print(f"      This suggests the model may be using shortcuts (not anatomical features).")
        
        # Show some examples
        print(f"\n   Example False Positives:")
        for i, (meta, analysis) in enumerate(stats['false_positives'][:5]):
            print(f"      {i+1}. Sample {meta['sample_idx']}, {meta['class_name']}: "
                  f"pred={meta['pred_prob']:.3f}, "
                  f"center_ratio={analysis['center_ratio']:.3f}, "
                  f"focused={'YES' if analysis['is_focused'] else 'NO'}")
    
    # Analyze true positives
    if stats['true_positives']:
        print(f"\nTRUE POSITIVES Analysis (gt=1, pred>0.3):")
        tp_focused = sum(1 for _, a in stats['true_positives'] if a['is_focused'])
        print(f"   Total TPs: {len(stats['true_positives'])}")
        print(f"   Focused on center: {tp_focused} ({tp_focused/len(stats['true_positives'])*100:.1f}%)")
        if tp_focused == len(stats['true_positives']):
            print(f"   All true positives focus on lung fields (good!)")
    
    print("\n" + "=" * 60)
    print("ASSESSMENT:")
    print("=" * 60)
    
    # Overall assessment
    total_focused = focused_count
    total_unfocused = unfocused_count
    focus_ratio = total_focused / len(heatmap_files) if heatmap_files else 0
    
    if focus_ratio > 0.8:
        print("EXCELLENT: Most heatmaps focus on lung fields (center regions)")
        print("   The model appears to be using anatomical features, not shortcuts.")
    elif focus_ratio > 0.6:
        print("GOOD: Most heatmaps focus on lung fields, but some show edge artifacts")
        print("   The model is mostly reliable, but may have some shortcut learning.")
    else:
        print("CONCERNING: Many heatmaps focus on edges/artifacts")
        print("   The model may be using shortcuts (background artifacts) instead of")
        print("   anatomical features. This suggests ROI extraction or training issues.")
    
    # Check for overconfidence
    all_metadata = [parse_filename(f.name) for f in heatmap_files]
    all_metadata = [m for m in all_metadata if m is not None]
    if all_metadata:
        avg_pred = np.mean([m['pred_prob'] for m in all_metadata])
    if avg_pred > 0.95:
        print(f"\nWARNING: Average prediction probability is very high ({avg_pred:.3f})")
        print("   This suggests the model may be overconfident or not well-calibrated.")


if __name__ == "__main__":
    main()

