# GeoCLIP Enhancement via LoRA on ImageEncoder

## 1. Problem Statement
Previous improvements focused on the **location_side** by introducing the *Coordinate-conditioned Attention* (SigmaSelector) and unfreezing the terminal layer of the `LocationEncoder`. While these changes improved spatial frequency modeling, the **image_side** still relies on a generic pre-trained CLIP extractor. 

The model currently lacks sensitivity to domain-specific street-view features—such as regional architectural styles, specific road signage, and local vegetation—which are critical for fine-grained localization. This stage aims to bridge the visual domain gap using **LoRA (Low-Rank Adaptation)**.

## 2. Implementation Plan
The goal is to adapt the `ImageEncoder` to street-view distributions without destroying its pre-trained global knowledge.

### 2.1 LoRA Configuration
*   **Target Modules**: LoRA will be injected into the `vision_model` of the CLIP backbone, specifically targeting the **Self-Attention** blocks.
    *   **Targets**: `q_proj`, `v_proj`.
*   **Hyperparameters**: 
    *   **Rank (r)**: 8 or 16 (optimized for the 11K dataset size).
    *   **Alpha**: 16 or 32 (typically $r \times 2$).
    *   **Dropout**: 0.05.

### 2.2 Training Strategy
To ensure stability and effective feature projection, we define the following training constraints:

**A. Trainable Parameters (requires_grad = True):**
1.  **LoRA Adapters**: The newly injected $A$ and $B$ matrices in the ViT.
2.  **Image Projection Head**: The `self.mlp` in `ImageEncoder` (768 $\rightarrow$ 768 $\rightarrow$ 512). *Note: This is crucial as it interprets the newly adapted CLIP features.*
3.  **SigmaSelector**: The coordinate-conditioned attention module.
4.  **LocationEncoder Tail**: The last layer of the `LocationEncoderCapsule`.

**B. Frozen Weights:**
*   All original weights of the CLIP Vision Transformer.
*   The backbone layers of the LocationEncoder.

**C. Optimization Protocol:**
*   **Low Learning Rate**: Use a base learning rate of $1 \times 10^{-4}$ for LoRA and MLP, and $5 \times 10^{-5}$ for location-related modules to ensure fine-tuning stability.
*   **Feasibility Check**: Perform a "Sanity Action" on a small subset (e.g., 500 images) to verify gradient flow and loss convergence before the full 11K run. 
*   **Visual Monitoring**: Generate `train_loss/val_loss vs. Epoch` plots to detect potential over-fitting early.

## 3. Expected Result
The fine-tuned model is expected to demonstrate superior "visual-to-spatial" alignment. 

### 3.1 Evaluation Metrics
We will evaluate the performance using the **Great-Circle Distance** between predicted and ground-truth coordinates:
*   **Recall@k (Accuracy)**: Success rate within thresholds of **1km, 25km, 200km, 750km and 2500km**.
*   **Median Error (km)**: The median distance error across the test set (a lower value indicates higher precision).

### 3.2 Baseline Comparison
*   **Baseline**: Current best model (SigmaSelector + Unfrozen last layer of LocationEncoder).
*   **Target**: The LoRA-enhanced model should achieve a significant improvement in **Recall@1km** and **Recall@25km**, demonstrating that the model has learned to recognize localized street-view cues.

**NOTICE**: All baseline checkpoints and logs must be strictly version-controlled (e.g., `baseline_v1.pth`) to ensure a rigorous "apples-to-apples" comparison with the new LoRA-enabled model.
