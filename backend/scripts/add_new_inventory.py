import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEW_INVENTORY_DIR = os.path.join(BASE_DIR, "backend", "new_inventory")
COLLECTION_NAME = "gold_ornaments"

# ==========================================================
# QDRANT CONNECTION
# ==========================================================
load_dotenv(os.path.join(BASE_DIR, '.env'))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    current_count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Current vectors in database: {current_count}")
except Exception as e:
    print(f"Error accessing collection '{COLLECTION_NAME}': {e}")
    print("Please make sure the collection exists (run migrate_to_qdrant.py first).")
    sys.exit(1)

# ==========================================================
# MODEL SETUP
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading ImageNet Pretrained ConvNeXt-Base...")
model = timm.create_model("convnext_base", pretrained=True, num_classes=0)
model = model.to(device)
model.eval()
print("Model Loaded Successfully!")

# ==========================================================
# PROCESS NEW INVENTORY
# ==========================================================
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
points_to_upsert = []

print(f"\nScanning for new images in: {NEW_INVENTORY_DIR}")
images_to_process = []
for root, dirs, files in os.walk(NEW_INVENTORY_DIR):
    for file in files:
        if file.lower().endswith(extensions):
            images_to_process.append(os.path.join(root, file))

if not images_to_process:
    print("No images found in the new_inventory folder. Exiting.")
    sys.exit(0)

print(f"Found {len(images_to_process)} images. Extracting features and uploading...")

start_id = current_count + 1
successful_uploads = 0

for i, image_path in enumerate(tqdm(images_to_process)):
    try:
        # Extract feature
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image_tensor)
            feature = feature.view(feature.size(0), -1)
            feature = F.normalize(feature, p=2, dim=1)

        vector = feature.squeeze().cpu().numpy().tolist()
        
        # Prepare for Qdrant
        abs_path = image_path.replace('\\', '/')
        point_id = start_id + i
        payload = {"image_url": abs_path}
        
        points_to_upsert.append(
            PointStruct(id=point_id, vector=vector, payload=payload)
        )
        successful_uploads += 1
        
    except Exception as e:
        print(f"\nError processing {image_path}: {e}")

if points_to_upsert:
    print("\nUploading to Qdrant...")
    batch_size = 50
    for j in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[j:j+batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    print(f"\nSuccess! Added {successful_uploads} new items to the database.")
else:
    print("\nFailed to process any images.")
