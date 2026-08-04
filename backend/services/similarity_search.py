import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from qdrant_client import QdrantClient

# Paths
MODEL_PATH = r"D:\Gold searching\models\best_model.pth"
# Use absolute path for Qdrant DB to ensure it's found regardless of where app.py is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QDRANT_DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "gold_ornaments"
NUM_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing ML Service on device:", device)
import torchvision.transforms.functional as F_vision

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        # Pad left, top, right, bottom
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"Connecting to Qdrant Cloud...")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def find_similar(image_path, top_k=20, offset=0, model_type="imagenet", category=None):
    """
    Given an image path, extracts features and queries Qdrant DB for cosine similarity.
    Note: model_type is kept for backwards compatibility but we only use imagenet vectors now.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model_imagenet(image)
        feature = feature.view(feature.size(0), -1)
        feature = F.normalize(feature, p=2, dim=1)
    
    query_vector = feature.squeeze().cpu().numpy().tolist()
    
    query_filter = None
    if category:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        )
    
    try:
        # Search Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            offset=offset
        )
    except Exception as e:
        print(f"Qdrant search error (Has the migration script been run?): {e}")
        return []
    
    results = []
    for rank, hit in enumerate(search_result, 1):
        # Qdrant returns scores. For cosine similarity, it's typically between -1 and 1.
        # Since we use vectors with non-negative components for images typically, it ranges 0-1.
        results.append({
            "rank": rank + offset,
            "similarity": float(hit.score),
            "image_url": hit.payload.get("image_url", "")
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
