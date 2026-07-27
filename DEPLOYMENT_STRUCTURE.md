# 🚀 Deployment Folder Structure

When you deploy this application to a production server (like AWS, Render, Heroku, or a local network server), you do **not** need the entire training dataset or all the training scripts (like `main.py`).

Here is the exact, minimized folder structure you need for deployment:

```text
Gold searching/
├── backend/
│   ├── api/
│   │   └── routes.py             # API endpoints
│   ├── services/
│   │   └── similarity_search.py  # ImageNet ML logic and classification check
│   ├── app.py                    # The Flask server entry point
│   ├── requirements.txt          # Python dependencies
│   └── uploads/                  # Temporary storage (created automatically)
│
├── frontend/                     
│   ├── index.html                # The web interface (Model select removed)
│   ├── style.css                 # Web styling
│   ├── script.js                 # Frontend API logic
│   └── bg.png                    # Background image for the UI
│
├── Embeddings/
│   ├── imagenet_image_features.npy  # ImageNet embeddings for similarity search
│   └── imagenet_image_paths.pkl     # Paths for the ImageNet embeddings
│
└── models/                       
    └── best_model.pth            # STILL REQUIRED! Used for the initial "Is this jewelry?" classification check
```

## Deployment Notes:
1. **Dataset:** You only need the `dataset/` folder if the frontend actually needs to display those images in the search results (which it does, since `imagenet_image_paths.pkl` points to `dataset/...`). If you want a truly minimal deployment, you would upload the dataset images to a cloud bucket (like AWS S3) and update the `.pkl` file with URL links instead of local paths. For now, keep the `dataset/` folder wherever the backend is running.
2. **Models:** Even though we switched to ImageNet for the similarity search, your backend still loads `best_model.pth` to perform the *Classification Check* (making sure the uploaded image is actually a ring/necklace/bangle and not a random object).
3. **Ignored Files:** You can safely leave behind `training_log.csv`, `main.py` scripts, and any original `.npy`/`.pkl` files in the `Embeddings/` folder that don't have `imagenet_` in their name.
