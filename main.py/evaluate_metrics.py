import pickle
import numpy as np
import os

# Paths
EMBEDDINGS_PATH = r"D:\Gold searching\Embeddings\image_features.npy"
IMAGE_PATHS_PATH = r"D:\Gold searching\Embeddings\image_paths.pkl"

def get_category(path):
    # e.g., D:\Gold searching\dataset\train\necklace\img.jpg
    # We want "necklace"
    parts = path.replace('\\', '/').split('/')
    # Usually the category is the parent of the image file
    return parts[-2].lower()

def evaluate():
    print("Loading features...")
    features = np.load(EMBEDDINGS_PATH)
    
    with open(IMAGE_PATHS_PATH, "rb") as f:
        paths = pickle.load(f)
        
    print(f"Loaded {len(paths)} items.")
    
    # Extract categories for ground truth
    categories = [get_category(p) for p in paths]
    
    # Evaluate a random sample of 500 images to save time, or do all if it's fast
    # Matrix multiplication for all 8000x8000 might take a few seconds
    print("Computing similarity matrix...")
    # Compute full similarity matrix: shape (N, N)
    # Since features are normalized, dot product is cosine similarity
    sim_matrix = np.dot(features, features.T)
    
    print("Calculating metrics...")
    N = len(paths)
    
    p_at_1 = 0
    p_at_5 = 0
    
    for i in range(N):
        # Sort indices by similarity descending
        # We ignore the very first one because it's the image itself (similarity = 1.0)
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        query_cat = categories[i]
        
        # Top 1 (excluding self at index 0)
        if categories[sorted_indices[1]] == query_cat:
            p_at_1 += 1
            
        # Top 5
        top_5_cats = [categories[idx] for idx in sorted_indices[1:6]]
        correct_in_top_5 = sum(1 for c in top_5_cats if c == query_cat)
        p_at_5 += correct_in_top_5 / 5.0
        
    avg_p1 = (p_at_1 / N) * 100
    avg_p5 = (p_at_5 / N) * 100
    
    print(f"Results across {N} images:")
    print(f"Category Precision@1: {avg_p1:.2f}%")
    print(f"Category Precision@5: {avg_p5:.2f}%")

if __name__ == "__main__":
    evaluate()
