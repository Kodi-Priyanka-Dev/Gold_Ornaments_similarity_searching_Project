import os
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# Paths
MODEL_PATH = r"D:\Gold searching\models\best_model.pth"
EMBEDDINGS_PATH = r"D:\Gold searching\Embeddings\image_features.npy"
IMAGE_PATHS_PATH = r"D:\Gold searching\Embeddings\image_paths.pkl"
IMAGENET_EMBEDDINGS_PATH = r"D:\Gold searching\Embeddings\imagenet_image_features.npy"
IMAGENET_PATHS_PATH = r"D:\Gold searching\Embeddings\imagenet_image_paths.pkl"
NUM_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing ML Service on device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Original ConvNeXt model into memory...")
model = timm.create_model("convnext_base", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.reset_classifier(0)
model.to(device)
model.eval()

print("Loading ImageNet ConvNeXt model into memory...")
model_imagenet = timm.create_model("convnext_base", pretrained=True, num_classes=0)
model_imagenet.to(device)
model_imagenet.eval()

print("Loading ConvNeXt classification model into memory...")
class_model = timm.create_model("convnext_base", pretrained=False, num_classes=NUM_CLASSES)
class_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
class_model.to(device)
class_model.eval()

CLASS_NAMES = [
    "Bangles",
    "Bracelets",
    "Earrings",
    "Hand Anklets",
    "Necklaces",
    "Rings"
]
CONFIDENCE_THRESHOLD = 0.70

print("Loading original dataset embeddings into memory...")
database_features = np.load(EMBEDDINGS_PATH)
with open(IMAGE_PATHS_PATH, "rb") as f:
    image_paths = pickle.load(f)

print("Loading ImageNet dataset embeddings into memory...")
if os.path.exists(IMAGENET_EMBEDDINGS_PATH):
    database_features_imagenet = np.load(IMAGENET_EMBEDDINGS_PATH)
    with open(IMAGENET_PATHS_PATH, "rb") as f:
        image_paths_imagenet = pickle.load(f)
else:
    database_features_imagenet = None
    image_paths_imagenet = None

print(f"ML Service ready. Loaded {len(image_paths)} original embeddings.")

def find_similar(image_path, top_k=20, model_type="original"):
    """
    Given an image path, extracts features and computes cosine similarity
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "imagenet" and database_features_imagenet is not None:
            feature = model_imagenet(image)
            db_features = database_features_imagenet
            db_paths = image_paths_imagenet
        else:
            feature = model(image)
            db_features = database_features
            db_paths = image_paths
            
        feature = feature.view(feature.size(0), -1)
        feature = F.normalize(feature, p=2, dim=1)
    
    query_feature = feature.squeeze().cpu().numpy()
    
    # Compute cosine similarity (dot product of normalized vectors)
    scores = np.dot(db_features, query_feature)
    
    # Sort descending
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        abs_path = db_paths[idx].replace('\\', '/')
        
        results.append({
            "rank": rank,
            "similarity": float(scores[idx]),
            "image_url": abs_path 
        })
        
    return results

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = class_model(image)
        probability = F.softmax(output, dim=1)
    confidence, prediction = torch.max(probability, dim=1)
    return confidence.item(), CLASS_NAMES[prediction.item()]
