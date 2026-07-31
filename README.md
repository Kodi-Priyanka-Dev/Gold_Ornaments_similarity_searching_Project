# 💍✨ Gold Ornament Similarity Search

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?logo=flask&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)

An AI-powered visual search engine designed to find similar gold jewelry from a large dataset in milliseconds. This project uses a deep learning model to extract image features, leveraging **Qdrant Vector Database** to compute cosine similarities. It wraps it all in a beautiful, responsive, and glassmorphism-styled web interface.

---

## ✨ Features

- **AI-Powered Visual Search:** Utilizes a state-of-the-art **ConvNeXt Base** model via `timm` to extract rich feature embeddings from images.
- **Blazing Fast Queries:** Embeddings are stored and queried using **Qdrant Vector Database**, making similarity search across thousands of images near-instantaneous.
- **Premium UI/UX:** A custom-built, vanilla HTML/CSS/JS frontend featuring drag-and-drop uploads, sleek animations, and a responsive glassmorphic design.
- **Luxurious Admin Dashboard:** A dedicated, premium black-and-gold admin interface for easily uploading and indexing new jewelry inventory.
- **Smart Results Filtering:** Intelligently separates results into "Exact Matches" and "Similar Images" for intuitive browsing.
- **Unified Server:** The Flask backend serves both the API and the web interfaces seamlessly on a single port.

---

## 🛠️ Technology Stack

- **Machine Learning:** PyTorch, `timm` (ConvNeXt Base), NumPy, PIL (Pillow), Torchvision
- **Backend & Database:** Python, Flask, Flask-CORS, Werkzeug, **Qdrant Vector Database**
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6)

---

## 📁 Project Structure

```text
Gold searching/
├── backend/
│   ├── api/
│   │   └── routes.py             # Flask API endpoints (upload & image serving)
│   ├── services/
│   │   └── similarity_search.py  # Core ML inference and Qdrant similarity logic
│   ├── uploads/                  # Temporary storage for query images
│   ├── app.py                    # Main Flask application entry point
│   └── requirements.txt          # Backend dependencies
├── frontend/                     
│   ├── index.html                # The main search web interface
│   ├── style.css                 # Custom glassmorphism styling
│   ├── script.js                 # Search drag-and-drop & API fetch logic
│   ├── admin.html                # Luxurious admin panel for uploading inventory
│   ├── admin.css                 # Admin panel specific styling
│   └── admin.js                  # Admin panel upload and progress tracking logic
├── qdrant_db/                    # Local Qdrant vector database storage
├── models/                       # Stored model weights (e.g., best_model.pth)
├── BUILD.md                      # Detailed Installation and Deployment Guide
└── README.md                     # This file
```

---

## 🚀 Setup, Installation, and Deployment

For comprehensive, step-by-step instructions on setting up the local environment, starting the application, and deploying it to a production server, please refer to the **[BUILD.md](BUILD.md)** guide.

### Quick Start (Development)
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `cd backend && python app.py`
3. Access Search: `http://localhost:5000/`
4. Access Admin Panel: `http://localhost:5000/admin.html`

---

## 💻 Usage

### Search for Jewelry
Drag and drop an image of a gold ornament into the main search upload zone, or click to browse. The UI will instantly display a preview, and the backend will process the image to return exact and similar matches from your dataset.

### Upload New Inventory
Navigate to the Admin Panel (`/admin.html`). Drag and drop images or select an entire folder of new gold ornaments. The AI will extract features and index them into Qdrant automatically.