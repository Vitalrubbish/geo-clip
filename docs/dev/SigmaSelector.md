# SigmaSelecgtor

## Problem Raising
In the current implementation, the LocationEncoder has 3 modules with $\sigma = [2^0, 2^4, 2^8]$. However, when forwarding the location features, the GeoCLIP simply add values from 3 modules together without any weight bias. This may cause serious problems. For example, when the picture is in streets in a city, it is better to use small $\sigma$ since the scene varies from block to block. When the picture is in a desert, mountains or ocean, it is better to use a larger $\sigma$.

## Goal
You need to:
1. Implement a `SigmaSelector` module, which receives location and returns a weight vector, telling the weight of each module.
2. Modify the forward mechanism of LocationEncoder, multiply the weight vector and `self._modules` to get the location_features.
3. Specially train this `SigmaSelector` with training data. 
4. Insert `SigmaSelector` and retest, comparing with the baseline.

## Implementation

### SigmaSelector
You need to define a new class `SigmaSelector` in `geoclip/model/location_encoder.py`, where there is an attetion-net in members of this class.

The attention-net is an `nn.Sequential`. Serving as a framework, the attention-net simply contains 4 layers:
- `Linear(m, 64)`
- `ReLU()`
- `Linear(64, n)`
- `Softmax()`
where $n$ represents dimension of `sigma` and `m` represents the dimension of `location`.

### Foward Mechanism of LocationEncoder
We do not simply add values returned from modules together without any weight bias. Now, with SigmaSelector, you can call `SigmaSelector` first to get the weight vector, and then multiply this weight vector with `self._modules` to get the location_features.

### Implement Training Function
In this part, you need to implement the train function `train_sigma_selector` so that it can train the `SigmaSelector`. 
- The training data has already downloaded into the local repo and has divided into train data and validate data with a ratio of `9:1`.
- The origin images are in `data/streetview_pano`, where there are 15K images in total. And locations (lons and lats) of these images are in `train_subset.csv` (for train data) and `val_subset.csv` (for validate data).
- Since we have limited computing resources, we take the training policy as follow:
    - We DO NOT TRAIN WEIGHTS OF OTHER MODULES! What we only need to train is our newly-designed `SigmaSelector`.
    - Therefore, you need to FREEZE other modules when we backward the gradient.
    - The weight of other modules is already in `geoclip/model/weights`, you can simply load it from there.

### Insert Module And Retest
In this part, we combine our `SigmaSelector` and existing modules together, and you need to write a script to test the performance on dataset `im2gps3k` and `test_subset.csv` in `streetview_pano` dataset. Similar to `scripts/eval_im2gps3k_pretrained.py`, you need to count the accuray with thresholds of `1km 25km 200km 750km 2500km`.

## Running
You need to train `SigmaSelector` first, and then retest the performance on test datasets.

## WARNING
In order to compare with baseline, whenever what modification you have made, you have to RESERVE baseline model. You can add parameters `dev = True(False)` to represent whether we test our new model or baseline model.