# Quick Test Guide for ROI Extraction

## Option 1: Test on Small Sample First (Recommended)

Test on just 100 images to verify quality before processing everything:

```powershell
# Create a test CSV with first 100 images
python -c "import pandas as pd; df = pd.read_csv('datasets/NIH Chest X-Rays Master Datasets/archive/Data_Entry_2017.csv'); df.head(100).to_csv('test_sample.csv', index=False)"

# Preprocess just 100 images
python scripts/preprocess_images.py --csv "test_sample.csv" --images-root "datasets/NIH Chest X-Rays Master Datasets/archive" --cache-dir "datasets/cache_test" --image-size 320 320 --labels Nodule Fibrosis
```

Then check the results in `datasets/cache_test/processed_images/`

## Option 2: Visualize Before Preprocessing

See what ROI extraction will do on random samples:

```powershell
python scripts/visualize_roi.py --num-samples 20
```

View results in:
- `roi_visualizations/roi_extraction_samples.png` (grid view)
- `roi_visualizations/individual_samples/` (individual comparisons)

## Option 3: Full Preprocessing

Process all images (takes 30-60 minutes):

```powershell
python scripts/preprocess_images.py --csv "datasets/NIH Chest X-Rays Master Datasets/archive/Data_Entry_2017.csv" --images-root "datasets/NIH Chest X-Rays Master Datasets/archive" --cache-dir "datasets/cache" --image-size 320 320 --labels Nodule Fibrosis
```

## What to Check

After preprocessing, verify:
1. ✅ Images are cropped to lung region (not full image)
2. ✅ Edge artifacts removed (hospital tags, borders)
3. ✅ Both lungs visible in most images
4. ✅ Images are square (320x320)
5. ✅ No obvious distortions or missing lung areas

## If Results Look Good

Once you verify the ROI extraction quality:
1. Delete old cache: `Remove-Item -Recurse -Force "datasets/cache"`
2. Run full preprocessing
3. Start training with cached images







