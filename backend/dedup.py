import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "gold_ornaments"

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

try:
    current_count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Total vectors before deduplication: {current_count}")
except Exception as e:
    print(f"Error accessing collection: {e}")
    sys.exit(1)

# Fetch all points using pagination
print("Fetching all points to find duplicates...")
all_points = []
offset = None
while True:
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    points, offset = response
    all_points.extend(points)
    if offset is None:
        break

print(f"Fetched {len(all_points)} points.")

# Track seen image URLs
seen_urls = set()
points_to_delete = []

for point in all_points:
    image_url = point.payload.get("image_url")
    if image_url in seen_urls:
        points_to_delete.append(point.id)
    else:
        seen_urls.add(image_url)

if points_to_delete:
    print(f"Found {len(points_to_delete)} duplicate points. Deleting...")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=points_to_delete
    )
    print("Deletion complete.")
else:
    print("No duplicates found.")

current_count = client.count(collection_name=COLLECTION_NAME).count
print(f"Total vectors after deduplication: {current_count}")
