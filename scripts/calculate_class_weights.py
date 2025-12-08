"""Calculate class weights for imbalanced dataset."""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

csv_path = Path("datasets/NIH Chest X-Rays Master Datasets/archive/Data_Entry_2017.csv")
label_columns = ["Nodule", "Fibrosis"]

metadata = pd.read_csv(csv_path)
total = len(metadata)

# Create binary labels
for label in label_columns:
    metadata[label] = metadata["Finding Labels"].str.contains(label, regex=False).astype(int)

print("Calculating class weights...")
print("="*60)

class_weights = []
for label in label_columns:
    positive = metadata[label].sum()
    negative = total - positive
    
    # Weight for positive class = total_negative / total_positive
    # Weight for negative class = 1.0 (baseline)
    weight_positive = negative / positive if positive > 0 else 1.0
    weight_negative = 1.0
    
    print(f"\n{label}:")
    print(f"  Positive samples: {positive:,}")
    print(f"  Negative samples: {negative:,}")
    print(f"  Weight for positive class: {weight_positive:.2f}")
    print(f"  Weight for negative class: {weight_negative:.2f}")
    
    class_weights.append(weight_positive)

print("\n" + "="*60)
print("RECOMMENDED CLASS WEIGHTS FOR CONFIG:")
print("="*60)
print(f"class_weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}]")
print(f"\nOr as integers (rounded):")
print(f"class_weights: [{int(class_weights[0])}, {int(class_weights[1])}]")











