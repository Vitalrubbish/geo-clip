# Testing Document
## All-Set Testing
1. Initial Model: GeoCLIP
- Eval Script: 

    ```bash
    python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector false
    ```
2. Baseline Model: SigmaSelector + unfrozen capsule heads.
- Train Script: 
    ```bash
    python scripts/train_sigma_selector.py \
    --mode full --epochs 10 --batch-size 32 \
    --unfreeze-capsule-head
    ```
- Eval Script: 
    ```bash
    python scripts/eval_sigma_selector.py \
    --dataset streetview_pano \
    --use-sigma-selector true \
    --selector-checkpoint outputs/sigma_selector/full_<timestamp>/selector_best.pth \
    --output-json data/streetview_pano/baseline_v1_eval.json
    ```
3. New Model: Baseline + LoRA on CLIP ViT last 6 layers + unfrozen image MLP
   - LoRA Configuration: r=8/16, alpha=16/32 (r×2), dropout=0.05
   - Target: q_proj, v_proj in ViT layers 18-23 (last 6 layers of ViT-L/14)
   - Trainable: LoRA adapters, image MLP, SigmaSelector, LocationEncoderCapsule heads
   - Frozen: CLIP ViT backbone (excluding LoRA), LocationEncoder backbone
   - Optimizer: Mixed precision (torch.cuda.amp), gradient checkpointing disabled
   - Learning Rates: 1e-4 for LoRA/MLP, 5e-5 for location modules
- Train Script:
    ```bash
    python scripts/train_lora.py \
    --mode full --epochs 10 --batch-size 32 \
    --lora-r 8 --lora-alpha 16 --lora-lr 1e-4 --location-lr 5e-5
    ```
- Eval Script:
    ```bash
    python scripts/eval_lora.py \
    --dataset streetview_pano \
    --checkpoint outputs/lora/full_<timestamp>/lora_best.pth
    ```
## Feasibility Check (both models)
1. Baseline
    ```bash
    python scripts/train_sigma_selector.py \
    --mode feasibility --epochs 10 --batch-size 16 \
    --unfreeze-capsule-head
    ```
2. New Model (900 train / 100 val subset)
    ```bash
    python scripts/train_lora.py \
    --mode feasibility --epochs 10 --batch-size 16 \
    --lora-r 8 --lora-alpha 16 --lora-lr 1e-4 --location-lr 5e-5
    ```

**Notice**: You can modify any parameters as you like.