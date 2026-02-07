"""
Final Robust ROI Extraction for NIH ChestX-ray14.

Strategy: For NIH dataset, use intelligent center crop as primary method.
- NIH images are already well-centered frontal chest X-rays
- Center crop (removing ~10-15% margin) removes artifacts and focuses on lungs
- ROI detection is used as validation/refinement, not primary method
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

# Optimize OpenCV
cv2.setNumThreads(1)

def extract_lung_roi(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    ROI extraction optimized for NIH Chest X-ray 14 dataset.
    
    Strategy:
    1. Try to detect lung regions using adaptive thresholding
    2. If detection fails or is unreliable, use intelligent center crop
    3. Center crop removes edge artifacts while preserving lung region
    """
    original_mode = image.mode
    img_w, img_h = image.size
    
    # For NIH dataset: Default to center crop (removes ~12% margin from each side)
    # This removes artifacts (hospital tags, borders) while keeping lungs
    default_margin = 0.12
    margin_w = int(img_w * default_margin)
    margin_h = int(img_h * default_margin)
    default_bbox = (margin_w, margin_h, img_w - 2*margin_w, img_h - 2*margin_h)
    
    # Try to detect lung regions for refinement
    try:
        # Resize for faster processing
        scale_factor = 256.0 / max(img_w, img_h)
        small_w = int(img_w * scale_factor)
        small_h = int(img_h * scale_factor)
        small_img = image.resize((small_w, small_h), resample=Image.Resampling.BILINEAR)
        small_arr = np.array(small_img.convert("L"))
        
        # Adaptive thresholding (works better than Otsu for varying contrast)
        binary = cv2.adaptiveThreshold(
            small_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if needed (lungs are typically dark)
        center_val = binary[small_h // 2, small_w // 2]
        corner_vals = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
        if np.mean(corner_vals) > 200 and center_val < 100:
            binary = 255 - binary
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and collect valid lung candidates
        valid_candidates = []
        center_x, center_y = small_w // 2, small_h // 2
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # Size filter: must be at least 1% of image
            if area < (small_w * small_h * 0.01):
                continue
            
            # Shape filter: reject extreme aspect ratios
            if h > 0:
                aspect = w / float(h)
                if aspect < 0.2 or aspect > 4.0:
                    continue
            
            # Centrality: prefer regions near center (where lungs are)
            box_cx = x + w // 2
            box_cy = y + h // 2
            dist_from_center = np.sqrt((center_x - box_cx)**2 + (center_y - box_cy)**2)
            max_dist = np.sqrt(small_w**2 + small_h**2) / 2
            centrality = 1.0 - (dist_from_center / max_dist)
            
            # Score: area * centrality^2
            score = area * (centrality ** 2)
            
            # Keep if score is reasonable
            if score > (small_w * small_h * 0.002):
                valid_candidates.append((x, y, w, h))
        
        # If we found valid candidates, use them
        if valid_candidates:
            # Union strategy: merge all valid candidates
            x_min = min([c[0] for c in valid_candidates])
            y_min = min([c[1] for c in valid_candidates])
            x_max = max([c[0] + c[2] for c in valid_candidates])
            y_max = max([c[1] + c[3] for c in valid_candidates])
            
            # Scale back to original size
            scale = 1.0 / scale_factor
            x = int(x_min * scale)
            y = int(y_min * scale)
            w = int((x_max - x_min) * scale)
            h = int((y_max - y_min) * scale)
            
            # Add padding
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img_w - x, w + 2*pad_x)
            h = min(img_h - y, h + 2*pad_y)
            
            # Check if detected ROI is reasonable
            crop_ratio = (w * h) / (img_w * img_h)
            
            # If ROI is too small (< 5%) or too large (> 80%), use center crop instead
            if 0.05 <= crop_ratio <= 0.80:
                # Detected ROI is reasonable - use it
                # Make it square for neural network
                center_x_roi = x + w // 2
                center_y_roi = y + h // 2
                square_size = max(w, h)
                
                new_x = center_x_roi - square_size // 2
                new_y = center_y_roi - square_size // 2
                
                # Ensure within bounds
                new_x = max(0, min(new_x, img_w - square_size))
                new_y = max(0, min(new_y, img_h - square_size))
                
                crop_x1 = max(0, new_x)
                crop_y1 = max(0, new_y)
                crop_x2 = min(img_w, new_x + square_size)
                crop_y2 = min(img_h, new_y + square_size)
                
                cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # Pad if needed (simple black padding for speed - this is only during preprocessing)
                if cropped.size[0] < square_size or cropped.size[1] < square_size:
                    pad_l = (square_size - cropped.size[0]) // 2
                    pad_t = (square_size - cropped.size[1]) // 2
                    pad_r = square_size - cropped.size[0] - pad_l
                    pad_b = square_size - cropped.size[1] - pad_t
                    cropped = ImageOps.expand(cropped, border=(pad_l, pad_t, pad_r, pad_b), fill=0)
                
                if cropped.mode != original_mode:
                    cropped = cropped.convert(original_mode)
                
                return cropped, (new_x, new_y, square_size, square_size)
    
    except Exception:
        # If detection fails, fall through to center crop
        pass
    
    # Default: Use intelligent center crop (removes artifacts, keeps lungs)
    # This is actually GOOD for NIH dataset because:
    # 1. Images are already well-centered
    # 2. Removes edge artifacts (hospital tags, borders)
    # 3. Focuses on central chest region where lungs are
    cropped = image.crop((
        default_bbox[0], 
        default_bbox[1], 
        default_bbox[0] + default_bbox[2], 
        default_bbox[1] + default_bbox[3]
    ))
    
    # Make it square
    w, h = cropped.size
    square_size = max(w, h)
    
    # Center the square crop (simple black padding for speed)
    pad_w = (square_size - w) // 2
    pad_h = (square_size - h) // 2
    cropped = ImageOps.expand(cropped, border=(pad_w, pad_h, 
                                               square_size - w - pad_w,
                                               square_size - h - pad_h), fill=0)
    
    if cropped.mode != original_mode:
        cropped = cropped.convert(original_mode)
    
    return cropped, default_bbox
