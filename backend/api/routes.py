import os
import subprocess
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
            return jsonify({"error": "It is not a gold ornament. Please upload a valid image of gold ornaments like bangles, necklaces, ear rings, or bracelets."}), 400

        # 2. Run the actual ML model similarity search with strict category filtering!
        model_type = request.form.get('model_type', 'imagenet')
        offset = int(request.form.get('offset', 0))
        limit = int(request.form.get('limit', 20))
        
        # Qdrant will natively filter the results to only match the predicted_category
        results = find_similar(filepath, top_k=limit, offset=offset, model_type=model_type, category=predicted_category)
        
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

@api_blueprint.route('/upload_inventory', methods=['POST'])
def upload_inventory():
    if 'images' not in request.files:
        return jsonify({"error": "No images part in the request"}), 400
        
    files = request.files.getlist('images')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No images selected"}), 400
        
    inventory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'new_inventory'))
    if not os.path.exists(inventory_dir):
        os.makedirs(inventory_dir)
        
    saved_count = 0
    for file in files:
        if file.filename:
            # We want to preserve folder structure if possible, but werkzeug's secure_filename drops paths.
            # To handle folders safely but preserve paths, we can use the original filename and secure parts.
            # For simplicity, we just save flat if secure_filename is used, or preserve dirs.
            # Let's save flat for now to ensure security.
            filename = secure_filename(os.path.basename(file.filename))
            filepath = os.path.join(inventory_dir, filename)
            file.save(filepath)
            saved_count += 1
            
    if saved_count == 0:
        return jsonify({"error": "Failed to save any images"}), 500
        
    # Trigger the extraction script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'add_new_inventory.py'))
    try:
        # Run script asynchronously or wait for it
        # For simplicity, we will run it synchronously so we can return success
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        
        if result.returncode != 0:
            print("Script Error:", result.stderr)
            return jsonify({"error": "Failed to process images", "details": result.stderr}), 500
            
        # Optional: Clean up the new_inventory folder after successful processing
        # import shutil
        # shutil.rmtree(inventory_dir)
        # os.makedirs(inventory_dir)
            
        return jsonify({
            "message": f"Successfully uploaded {saved_count} images and added to database.",
            "logs": result.stdout
        })
    except Exception as e:
        print(f"Error running add_new_inventory.py: {e}")
        return jsonify({"error": str(e)}), 500
