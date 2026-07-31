import urllib.request
import urllib.error
import json
import os

URL = 'http://localhost:5000/api/search'
IMAGE_PATH = r"d:\Gold searching\backend\new_inventory\Screenshot_2026-07-27_151313.png"

try:
    # A simple multipart/form-data upload using urllib is tedious, so we'll just read from the local Qdrant directly or use the `backend.services.similarity_search` module directly.
    import sys
    sys.path.append(r"d:\Gold searching\backend")
    from services.similarity_search import find_similar
    results = find_similar(IMAGE_PATH, top_k=10)
    print(f"Got {len(results)} results")
    for i, res in enumerate(results):
        print(f"{i+1}. Sim: {res['similarity']:.6f} - URL: {res['image_url']}")
except Exception as e:
    print("Exception:", e)
