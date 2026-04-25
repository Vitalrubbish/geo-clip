import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader
from geoclip.train.eval import eval_images
from torch.utils.data import DataLoader

# 1. Prepare dummy "small test set"
print("1. Preparing small test dataset (Baseline Data)...")
os.makedirs("data/images", exist_ok=True)
dummy_data = [
    {"IMG_FILE": "eiffel.jpg", "LAT": 40.7128, "LON": -74.0060},
    {"IMG_FILE": "tokyo.jpg", "LAT": 35.6895, "LON": 139.6917},
    {"IMG_FILE": "london.jpg", "LAT": 51.5074, "LON": -0.1278},
]

for item in dummy_data:
    img_path = os.path.join("data/images", item["IMG_FILE"])
    img = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
    img.save(img_path)

df = pd.DataFrame(dummy_data)
df.to_csv("data/test_subset.csv", index=False)

# 2. Loading Pre-trained Model (Baseline Model)
print("2. Loading GeoCLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCLIP(from_pretrained=True).to(device)
# ensure gallery is on device
model.gps_gallery = model.gps_gallery.to(device)

# 3. Simulate Core Trick: Feature Extraction
print("3. Pre-extracting visual features for fast evaluation (as suggested in plan.md)...")
extracted_features = {}
model.eval()

with torch.no_grad():
    for item in dummy_data:
        img_path = os.path.join("data/images", item["IMG_FILE"])
        img = Image.open(img_path).convert('RGB')
        img_tensor = model.image_encoder.preprocess_image(img).to(device)
        feature = model.image_encoder(img_tensor)
        extracted_features[item["IMG_FILE"]] = feature.cpu()
        
torch.save(extracted_features, "data/baseline_features.pt")
print("   Features saved to data/baseline_features.pt")

# 4. Run standard Baseline Evaluation 
print("4. Running standard evaluation script...")
class DummyGeoDataLoader(DataLoader):
    def __init__(self, df, device):
        self.data = df
        self.device = device
    def __iter__(self):
        imgs = []
        gps = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join("data/images", row["IMG_FILE"])
            img = Image.open(img_path).convert('RGB')
            img_tensor = model.image_encoder.preprocess_image(img).squeeze(0)
            imgs.append(img_tensor)
            gps.append((row["LAT"], row["LON"]))
        
        yield torch.stack(imgs).to(self.device), torch.tensor(gps).to(self.device)
    def __len__(self):
        return 1

dataloader = DummyGeoDataLoader(df, device)

accuracy_results = eval_images(dataloader, model, device=device)
print("\n--- Baseline Results ---")
for k, v in accuracy_results.items():
    print(f"{k}: {v:.4f}")
