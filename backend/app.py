from flask import Flask, jsonify
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from api.routes import api_blueprint
import os

# Serve static files directly from the frontend folder
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')

CORS(app)

app.register_blueprint(api_blueprint, url_prefix='/api')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/admin')
def serve_admin():
    return send_from_directory(app.static_folder, 'admin.html')

if __name__ == '__main__':
    print("---------------------------------------")
    print(" Running with model: convnext_base")
    print("---------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=True)
