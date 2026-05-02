"""
Flask application for Fake Image Detection with Explainable AI
Main entry point for the backend server
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for frontend communication
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
device = None

def init_model():
    """Initialize the ML model"""
    global model, device
    try:
        import torch
        from torchvision import models
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load ResNet50
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)  # Binary classification
        
        # Try to load checkpoint if exists
        checkpoint_path = 'backend/models/checkpoint.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded model checkpoint from {checkpoint_path}")
        else:
            print("⚠ No checkpoint found. Using ImageNet pretrained weights.")
            print("  Note: Model accuracy will be lower. Train with: python train.py")
        
        model.to(device)
        model.eval()
        print("✓ Model initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        raise

# ==================== API Routes ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'None'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API info"""
    return jsonify({
        'name': 'Hybrid Fake Image Detector API',
        'version': '1.0.0',
        'description': 'Detect AI-generated vs real images with explainable AI',
        'endpoints': {
            'health': 'GET /health',
            'detect': 'POST /api/detect',
            'documentation': 'GET /api/docs'
        },
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'device': str(device)
    }), 200

@app.route('/api/docs', methods=['GET'])
def docs():
    """API documentation endpoint"""
    return jsonify({
        'endpoints': {
            'POST /api/detect': {
                'description': 'Detect if image is real or AI-generated',
                'parameters': {
                    'image': 'Image file (multipart/form-data)'
                },
                'response': {
                    'prediction': 'Real or AI-generated',
                    'confidence': 'Confidence score (0-1)',
                    'explanation': 'Human-readable explanation',
                    'images': {
                        'original': 'Base64 encoded original image',
                        'ela': 'Base64 encoded ELA visualization',
                        'fft': 'Base64 encoded FFT visualization',
                        'grad_cam': 'Base64 encoded Grad-CAM heatmap'
                    }
                }
            }
        }
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint
    Accepts image file and returns prediction with explanations
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': f'Invalid file type. Allowed: {allowed_extensions}'}), 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + file.filename)
        file.save(temp_path)
        
        try:
            # Import detection function
            from backend_detection import process_image
            
            # Process image and get results
            result = process_image(temp_path, model, device)
            
            return jsonify(result), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500

# ==================== Error Handlers ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 10MB'}), 413

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request error"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({
        'error': 'Internal server error',
        'details': str(error)
    }), 500

# ==================== Main ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Fake Image Detector - Backend Server")
    print("="*50 + "\n")
    
    # Initialize model
    print("Initializing model...")
    init_model()
    
    print("\nStarting Flask server...")
    print("Server running at http://localhost:5000")
    print("API docs at http://localhost:5000/api/docs")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
