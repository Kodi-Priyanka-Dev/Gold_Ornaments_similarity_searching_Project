import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = r"D:\Gold searching\dataset"
MODEL_PATH = r"D:\Gold searching\models\best_model.pth"
OUTPUT_PATH = r"D:\Gold searching\Embeddings"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

import torchvision.transforms.functional as F_vision

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = (
            int((max_wh - w) / 2),
            int((max_wh - h) / 2),
            max_wh - w - int((max_wh - w) / 2),
            max_wh - h - int((max_wh - h) / 2)
        )
        return F_vision.pad(image, padding, fill=255, padding_mode='constant')

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# LOAD MODEL
# ==========================================================

NUM_CLASSES = 4   # Change according to your dataset

model = timm.create_model(
    "convnext_base",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Remove classifier and keep only feature extractor
model.reset_classifier(0)

model = model.to(device)
model.eval()

print("Model Loaded Successfully!")

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        # Extract embedding
        feature = model(image)

        # Flatten
        feature = feature.view(feature.size(0), -1)

        # L2 Normalization
        feature = F.normalize(feature, p=2, dim=1)

    return feature.squeeze().cpu().numpy()

# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ==========================================================
# EXTRACT FEATURES
# ==========================================================

total_images = 0
all_embeddings = []
all_image_paths = []

print("\nStarting Feature Extraction...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in tqdm(files):

        if file.lower().endswith(extensions):

            image_path = os.path.join(root, file)

            try:
                feature = extract_feature(image_path)
                
                all_embeddings.append(feature)
                all_image_paths.append(image_path)

                total_images += 1

            except Exception as e:
                print(f"Error processing {image_path}")
                print(e)

print("\nSaving embeddings and paths...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

if all_embeddings:
    embeddings_array = np.array(all_embeddings)
    np.save(os.path.join(OUTPUT_PATH, "image_features.npy"), embeddings_array)
    
    with open(os.path.join(OUTPUT_PATH, "image_paths.pkl"), "wb") as f:
        pickle.dump(all_image_paths, f)
else:
    embeddings_array = np.array([])
    print("No embeddings were generated!")

print("\n====================================")
print(f"Total Images Processed : {total_images}")
if all_embeddings:
    print(f"Embeddings Shape       : {embeddings_array.shape}")
print(f"Saved At               : {OUTPUT_PATH}")
print("L2 Normalization Applied Successfully")
print("Feature Extraction Completed")
print("\nFiles Saved:")
print("embeddings/image_features.npy")
print("embeddings/image_paths.pkl")
print("====================================")