import os
import time
import json
import random
import glob
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file.")
    print("Please get a free API key from https://aistudio.google.com/ and add it to your .env file like this:")
    print("GEMINI_API_KEY=\"your_key_here\"")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# Using gemini-1.5-flash as it is much faster and cheaper for bulk image tagging than Pro.
model = genai.GenerativeModel('gemini-1.5-flash')

PROMPT = """
You are an expert jewelry appraiser and cataloger. Look at this image of a gold ornament.
Output a valid JSON object with exactly these fields (use "Unknown" if you cannot determine it):
{
    "sub_category": "e.g., Jhumka, Stud, Drop, Choker, Chain, Temple, Kundan, Solitaire, etc.",
    "metal": "e.g., Yellow Gold, Rose Gold, White Gold, Antique Gold, Silver",
    "stone_type": "e.g., Pearl, Diamond, Ruby, Emerald, Kundan, None",
    "collection": "e.g., Bridal, Casual, Traditional, Modern, Temple"
}
Output ONLY the raw JSON object. Do not include markdown formatting or backticks.
"""

def generate_metadata(image_path):
    try:
        img = Image.open(image_path)
        # Resize to save bandwidth and speed up API call
        img.thumbnail((512, 512))
        
        response = model.generate_content([PROMPT, img])
        raw_text = response.text.strip()
        
        # Clean up any markdown blocks if the model mistakenly added them
        if raw_text.startswith('```json'):
            raw_text = raw_text[7:]
        if raw_text.startswith('```'):
            raw_text = raw_text[3:]
        if raw_text.endswith('```'):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def run_full_batch():
    output_file = os.path.join(BASE_DIR, "backend", "data", "rich_metadata.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing progress to resume if interrupted
    results = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                results = json.load(f)
                print(f"Resumed from {output_file} with {len(results)} already processed.")
            except json.JSONDecodeError:
                pass

    print("Gathering images from dataset...")
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    
    all_images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_images.append(os.path.join(root, file).replace('\\', '/'))
                
    images_to_process = [img for img in all_images if img not in results]
    
    print(f"Found {len(all_images)} total images. {len(images_to_process)} remaining to process.")
    
    if not images_to_process:
        print("All images have already been processed!")
        return

    print("\nStarting Gemini Auto-Tagging...")
    
    try:
        for i, img_path in enumerate(tqdm(images_to_process)):
            meta = generate_metadata(img_path)
            if meta:
                meta["image_id"] = os.path.basename(img_path)
                results[img_path] = meta
                
                # Save progress continuously every 10 images
                if i % 10 == 0:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4)
                        
            # Sleep 4 seconds to respect Gemini Free Tier 15 RPM limit
            time.sleep(4)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")
    finally:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved {len(results)} total tags to {output_file}.")

if __name__ == "__main__":
    run_full_batch()
