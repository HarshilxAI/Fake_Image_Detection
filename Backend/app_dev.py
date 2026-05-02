"""
Simplified Backend App - Without Heavy Dependencies
For initial testing and development
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== PLACEHOLDER FUNCTIONS ====================
# These will be replaced with actual model inference later

def mock_predict(image_path):
    """Mock prediction function"""
    # This will be replaced with actual model inference
    return {
        'prediction': 'Real',
        'confidence': 0.87
    }

def image_to_base64(pil_image):
    """Convert PIL Image to base64"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': False,
        'device': 'cpu',
        'version': '1.0.0-dev'
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'Hybrid Fake Image Detector API',
        'version': '1.0.0',
        'description': 'Detect AI-generated vs real images',
        'endpoints': {
            'health': 'GET /health',
            'detect': 'POST /api/detect',
            'docs': 'GET /api/docs'
        },
        'status': 'development'
    }), 200

@app.route('/api/docs', methods=['GET'])
def docs():
    """API documentation"""
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
                        'original': 'Base64 encoded image',
                        'ela': 'Error Level Analysis',
                        'fft': 'FFT visualization',
                        'grad_cam': 'Grad-CAM heatmap'
                    }
                }
            }
        }
    }), 200

@app.route('/api/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate extension
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + file.filename)
        file.save(temp_path)
        
        try:
            # Load image
            pil_image = Image.open(temp_path).convert('RGB')
            
            # Get mock prediction
            pred = mock_predict(temp_path)
            
            # Return response
            result = {
                'success': True,
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'confidence_percentage': round(pred['confidence'] * 100, 2),
                'explanation': f"This image appears to be {pred['prediction'].lower()}. Model confidence: {round(pred['confidence']*100, 1)}%",
                'images': {
                    'original': image_to_base64(pil_image),
                    'ela': image_to_base64(pil_image),
                    'fft': image_to_base64(pil_image),
                    'grad_cam': image_to_base64(pil_image)
                }
            }
            
            return jsonify(result), 200
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large (max 10MB)'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FAKE IMAGE DETECTOR - BACKEND SERVER (Development Mode)")
    print("="*70 + "\n")
    
    print("Starting Flask server...")
    print("Server running at: http://localhost:5000")
    print("API docs at: http://localhost:5000/api/docs")
    print("Health check: http://localhost:5000/health")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
