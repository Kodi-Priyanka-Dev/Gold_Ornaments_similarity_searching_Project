import os
import sys
import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Adjust paths relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")

IMAGENET_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "imagenet_image_features.npy")
IMAGENET_PATHS_PATH = os.path.join(EMBEDDINGS_DIR, "imagenet_image_paths.pkl")
QDRANT_DB_PATH = os.path.join(BASE_DIR, "qdrant_db")

COLLECTION_NAME = "gold_ornaments"
VECTOR_SIZE = 1024

def extract_category(path):
    path_lower = path.lower()
    if 'bangle' in path_lower: return 'Bangles'
    if 'bracelet' in path_lower: return 'Bracelets'
    if 'earring' in path_lower or 'ear ring' in path_lower: return 'Earrings'
    if 'anklet' in path_lower: return 'Hand Anklets'
    if 'necklace' in path_lower: return 'Necklaces'
    if 'ring' in path_lower: return 'Rings'
    return 'Unknown'

def migrate():
    print(f"Loading vectors from: {IMAGENET_EMBEDDINGS_PATH}")
    if not os.path.exists(IMAGENET_EMBEDDINGS_PATH):
        print("Error: Embeddings not found. Please ensure feature extraction has been run.")
        sys.exit(1)
        
    features = np.load(IMAGENET_EMBEDDINGS_PATH)
    
    print(f"Loading paths from: {IMAGENET_PATHS_PATH}")
    with open(IMAGENET_PATHS_PATH, "rb") as f:
        paths = pickle.load(f)
        
    total_images = len(paths)
        
    # Load rich metadata if it exists
    rich_metadata_path = os.path.join(BASE_DIR, "backend", "data", "rich_metadata.json")
    rich_meta = {}
    if os.path.exists(rich_metadata_path):
        print(f"Loading rich metadata from {rich_metadata_path}...")
        with open(rich_metadata_path, "r") as f:
            rich_meta = json.load(f)
            
    print(f"Loaded {total_images} records.")
    
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    print(f"Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
    
    # Check if collection exists
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
        
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    
    from qdrant_client.http.models import PayloadSchemaType
    print("Creating payload indexes for strict filtering...")
    
    # Create indexes for all searchable fields
    index_fields = ["category", "sub_category", "metal", "stone_type", "collection"]
    for field in index_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            print(f"Warning: Failed to create index for {field}: {e}")
    
    print("Uploading points to Qdrant in batches...")
    batch_size = 100
    for i in range(0, total_images, batch_size):
        end_idx = min(i + batch_size, total_images)
        
        points = []
        for j in range(i, end_idx):
            point_id = j + 1
            # Normalizing the path separators for consistency
            abs_path = paths[j].replace('\\', '/')
            
            # Base Payload
            cat = extract_category(abs_path)
            payload = {"image_url": abs_path, "category": cat}
            
            # Inject Generative AI Rich Metadata if it exists for this image
            if abs_path in rich_meta:
                ai_data = rich_meta[abs_path]
                payload["sub_category"] = ai_data.get("sub_category", "Unknown")
                payload["metal"] = ai_data.get("metal", "Unknown")
                payload["stone_type"] = ai_data.get("stone_type", "Unknown")
                payload["collection"] = ai_data.get("collection", "Unknown")
            
            # The features array is likely already normalized by torch.F.normalize
            vector = features[j].tolist()
            
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
            
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"  Uploaded batch {i} to {end_idx}...")
        
    print(f"Migration complete! Uploaded {total_images} records into Qdrant.")
    
    print("\n---------------------------------------------------------")
    print("Core dataset migration complete! Now restoring new inventory items...")
    add_script = os.path.join(os.path.dirname(__file__), "add_new_inventory.py")
    if os.path.exists(add_script):
        import subprocess
        subprocess.run([sys.executable, add_script])
    else:
        print("add_new_inventory.py not found.")

if __name__ == "__main__":
    migrate()
