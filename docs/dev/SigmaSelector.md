# SigmaSelector Design (Executable Spec)

## 1. Problem Statement
Current `LocationEncoder` uses 3 branches with sigma values `[2**0, 2**4, 2**8]` and sums outputs equally.
This document introduces a learnable `SigmaSelector` to produce adaptive branch weights.

Important scope clarification:
1. In this design, selector input is only GPS location.
2. Therefore, adaptation is location-conditioned, not image-conditioned.

## 2. Objectives
1. Add `SigmaSelector` in `geoclip/model/location_encoder.py`.
2. Replace equal-sum fusion with weighted fusion.
3. Train only `SigmaSelector` while freezing all other parameters.
4. Keep baseline path available for A/B comparison.
5. Evaluate on both Im2GPS3K and StreetView Pano test subsets.

## 3. Model Specification

### 3.1 SigmaSelector
Define class `SigmaSelector(nn.Module)` in `geoclip/model/location_encoder.py`.

Input and output:
1. Input `location`: shape `(B, m)`, where `m=2` after Equal Earth projection.
2. Output `weights`: shape `(B, n)`, where `n=len(sigma)`.
3. `weights` must satisfy row-wise sum equals 1.

Network:
1. `Linear(m, 64)`
2. `ReLU()`
3. `Linear(64, n)`
4. `Softmax(dim=-1)`

Initialization requirement:
1. Initialize the last linear layer with zeros for both weight and bias.
2. This guarantees pre-training output is uniform `[1/n, ..., 1/n]`.

### 3.2 Weighted Fusion in LocationEncoder
Do not multiply tensor by `self._modules` directly.
Correct fusion is:

`f_loc = sum_i w_i(location) * f_i(location)`

Implementation contract:
1. Keep existing branch modules (`LocEnc0`, `LocEnc1`, `LocEnc2`).
2. Compute all branch features with shape `(B, 512)`.
3. Stack to `(B, n, 512)`.
4. Broadcast weights `(B, n, 1)` and sum over axis `n`.

### 3.3 Baseline Compatibility
Add explicit switch instead of `dev=True/False`.

Recommended API:
1. `LocationEncoder(..., use_sigma_selector=False)`
2. `GeoCLIP(..., use_sigma_selector=False)`

Behavior:
1. `False`: original equal-sum baseline.
2. `True`: sigma-selector weighted fusion.

## 4. Training Specification

### 4.1 Data
StreetView Pano files:
1. Train CSV: `data/streetview_pano/train_subset.csv`
2. Val CSV: `data/streetview_pano/val_subset.csv`
3. Test CSV: `data/streetview_pano/test_subset.csv`
4. Image root: `data/streetview_pano/images`

CSV contract must match `GeoDataLoader`:
1. Column `IMG_FILE`
2. Column `LAT`
3. Column `LON`

### 4.2 Optimization Policy
Train only `SigmaSelector`.

Freeze list:
1. `ImageEncoder` all parameters
2. `LocationEncoder` branch capsules (`LocEnc*`) parameters
3. `logit_scale`

Trainable list:
1. `SigmaSelector` parameters only

Loss:
1. Use the same contrastive loss path as current GeoCLIP training.
2. Keep GPS queue logic unchanged.

### 4.3 New Training Entry
Implement a new training function/script, for example:
1. Function: `train_sigma_selector(...)`
2. Script: `scripts/train_sigma_selector.py`

Required outputs:
1. Best selector checkpoint by val loss
2. Per-epoch train loss
3. Per-epoch val loss
4. JSON log file with hyperparameters and metrics

## 5. Feasibility Check (900/100)

### 5.1 Subset Sampling
Create script, for example `scripts/sample_streetview_subset.py`.

Requirements:
1. Randomly sample 1000 unique rows from `train_subset.csv + val_subset.csv` pool.
2. Split into 900 train and 100 val.
3. Use fixed random seed for reproducibility.
4. Save as two CSV files under `data/streetview_pano/feasibility`.

### 5.2 Feasibility Training
1. Train 20 epochs on 900/100 split.
2. Record train/val loss each epoch.
3. Plot and save loss curve figure.

Acceptance criteria:
1. Val loss trend is non-divergent.
2. Final val loss is lower than epoch-1 val loss.

## 6. Full Training
1. Train on official train split and validate on official val split.
2. Save final selector checkpoint and best selector checkpoint.
3. Save full training log and loss curves.

## 7. Evaluation Specification

### 7.1 Metrics
Report `acc_{d}_km` for:
1. 1 km
2. 25 km
3. 200 km
4. 750 km
5. 2500 km

### 7.2 Datasets
Run both:
1. Im2GPS3K test set
2. StreetView Pano test set

### 7.3 Comparison
For each dataset, report:
1. Baseline metrics
2. SigmaSelector metrics
3. Delta (SigmaSelector - Baseline)

Output format:
1. JSON result file
2. Markdown table for quick reading

## 8. Runbook

Environment notes:
1. Activate conda env before all commands.
2. Build updated module before running training/eval.

Suggested command sequence:
1. `conda activate geoclip`
2. `pip install -e .`
3. `python scripts/sample_streetview_subset.py`
4. `python scripts/train_sigma_selector.py --mode feasibility --epochs 20`
5. `python scripts/train_sigma_selector.py --mode full`
6. `python scripts/eval_sigma_selector.py --dataset im2gps3k --use-sigma-selector true`
7. `python scripts/eval_sigma_selector.py --dataset streetview_pano --use-sigma-selector true`
8. `python scripts/eval_sigma_selector.py --dataset im2gps3k --use-sigma-selector false`
9. `python scripts/eval_sigma_selector.py --dataset streetview_pano --use-sigma-selector false`

## 9. Deliverables Checklist
1. Code changes in `location_encoder.py` and model wiring.
2. Selector-only training script.
3. Feasibility subset sampling script.
4. Evaluation script supporting baseline and selector modes.
5. Loss curves and JSON logs.
6. Final comparison table for both datasets.

## 10. Risk Notes
1. Because selector is GPS-only, it cannot directly adapt to image content.
2. If gains are weak, consider a future variant where selector also takes image features.
3. Keep baseline path intact for fair comparison and rollback safety.