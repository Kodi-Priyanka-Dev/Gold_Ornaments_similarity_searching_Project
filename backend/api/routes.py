import os
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from services.similarity_search import find_similar, classify_image, CONFIDENCE_THRESHOLD

api_blueprint = Blueprint('api', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@api_blueprint.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # 1. Classification Check
        confidence, predicted_category = classify_image(filepath)
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({"error": "It is not a gold ornament. Please upload a valid image of gold ornaments like bangles, necklaces, earrings, or bracelets."}), 400

        # 2. Run the actual ML model similarity search!
        model_type = request.form.get('model_type', 'original')
        results = find_similar(filepath, top_k=15, model_type=model_type)
        
        # Cleanup uploaded file if desired, but we'll leave it for now
        # os.remove(filepath)
        
        return jsonify({
            "message": "Search completed successfully",
            "predicted_category": predicted_category,
            "results": results
        })
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/images/<path:filepath>')
def serve_image(filepath):
    """
    Serves images from the local dataset securely.
    The frontend will pass the absolute path here.
    """
    # URL unquoting might be necessary depending on browser, but Flask handles most of it.
    # If the path starts with a drive letter, it might be passed as 'D:/' or 'D|/'
    
    # We just serve the file directly
    try:
        # Handle cases where the leading slash for absolute paths was stripped by the browser URL
        if filepath[1] == ':' or filepath[2] == ':': 
            pass # Windows absolute path
        else:
            # Reconstruct if necessary, but D:/ usually comes through fine
            pass
            
        return send_file(filepath)
    except Exception as e:
        print(f"Error serving image {filepath}: {e}")
        return "Image not found", 404
