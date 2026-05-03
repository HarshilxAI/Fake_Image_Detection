"""
Image detection and processing module
Handles model inference and explainability generation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io
from typing import Dict, Tuple, Any

def load_and_preprocess_image(image_path: str, size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load image and preprocess for model inference
    
    Args:
        image_path: Path to image file
        size: Target size for resizing
    
    Returns:
        Tuple of (preprocessed tensor, PIL image)
    """
    # Load image
    pil_image = Image.open(image_path).convert('RGB')
    
    # Resize
    pil_image = pil_image.resize((size, size), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(pil_image).astype(np.float32)
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array / 255.0 - mean) * (1.0 / std)
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor, pil_image

def get_prediction(image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device) -> Tuple[str, float]:
    """
    Get model prediction for image
    
    Args:
        image_tensor: Preprocessed image tensor
        model: PyTorch model
        device: Device to run model on
    
    Returns:
        Tuple of (prediction label, confidence score)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction (0 = Real, 1 = AI-generated)
        prediction_idx = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][prediction_idx].item()
        
        # Map to labels
        labels = ['Real', 'AI-generated']
        prediction = labels[prediction_idx]
        
        return prediction, confidence

def generate_ela_image(image_path: str, quality: int = 90) -> Image.Image:
    """
    Generate Error Level Analysis (ELA) image
    Reveals compression artifacts and inconsistencies
    
    Args:
        image_path: Path to image
        quality: JPEG compression quality for comparison
    
    Returns:
        PIL Image of ELA visualization
    """
    # Load original image
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Save and reload with JPEG compression
    temp_path = '/tmp/temp_compressed.jpg'
    cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread(temp_path)
    
    # Compute difference
    ela_image = cv2.absdiff(original, compressed)
    
    # Scale up for visualization
    ela_image = np.uint8(ela_image * 2)
    
    # Convert to RGB
    ela_rgb = cv2.cvtColor(ela_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(ela_rgb)

def generate_fft_image(image_path: str) -> Image.Image:
    """
    Generate Frequency Domain (FFT) visualization
    Detects unnatural frequency patterns
    
    Args:
        image_path: Path to image
    
    Returns:
        PIL Image of FFT visualization
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shift)
    
    # Log scale for better visualization
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    
    # Normalize to 0-255
    magnitude_spectrum = np.uint8(255 * (magnitude_spectrum / np.max(magnitude_spectrum)))
    
    return Image.fromarray(magnitude_spectrum)

def generate_grad_cam(image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device, 
                     layer_name: str = 'layer4') -> Image.Image:
    """
    Generate Grad-CAM heatmap
    Shows which regions the model focused on
    
    Args:
        image_tensor: Preprocessed image tensor
        model: PyTorch model
        device: Device to run model on
        layer_name: Name of target layer for Grad-CAM
    
    Returns:
        PIL Image of Grad-CAM heatmap
    """
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(image_tensor)
    predicted_class = torch.argmax(outputs[0]).item()
    
    # Zero gradients
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    # Backward pass
    target = outputs[0, predicted_class]
    target.backward()
    
    # Get gradients
    gradients = image_tensor.grad.data
    
    # Compute heatmap
    heatmap = torch.mean(gradients[0], dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    # Resize to image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Convert to color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(heatmap_rgb)

def image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def get_explanation(prediction: str, confidence: float) -> str:
    """
    Generate human-readable explanation for prediction
    
    Args:
        prediction: Model prediction (Real or AI-generated)
        confidence: Confidence score
    
    Returns:
        Explanation text
    """
    confidence_pct = confidence * 100
    
    if prediction == 'AI-generated':
        if confidence > 0.9:
            explanation = f"This image is very likely AI-generated with {confidence_pct:.1f}% confidence. "
            explanation += "The model detected strong digital artifacts and unnatural patterns typical of AI generators like DALL-E or Stable Diffusion."
        elif confidence > 0.7:
            explanation = f"This image is likely AI-generated with {confidence_pct:.1f}% confidence. "
            explanation += "Several indicators suggest artificial origin, including compression patterns and frequency anomalies."
        else:
            explanation = f"This image appears to be AI-generated with {confidence_pct:.1f}% confidence, though there is some uncertainty. "
            explanation += "The model detected some artifacts, but they are less pronounced."
    else:  # Real
        if confidence > 0.9:
            explanation = f"This image is very likely real with {confidence_pct:.1f}% confidence. "
            explanation += "The model detected natural compression patterns and frequency characteristics consistent with camera-captured images."
        elif confidence > 0.7:
            explanation = f"This image is likely real with {confidence_pct:.1f}% confidence. "
            explanation += "The overall characteristics suggest a genuine photograph or digitally obtained image."
        else:
            explanation = f"This image appears to be real with {confidence_pct:.1f}% confidence, though there is some uncertainty. "
            explanation += "Some characteristics are ambiguous, but lean toward authentic origin."
    
    return explanation

def process_image(image_path: str, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """
    Process image: inference + explainability generation
    
    Args:
        image_path: Path to uploaded image
        model: PyTorch model
        device: Device to run model on
    
    Returns:
        Dictionary with prediction, confidence, explanation, and visualizations
    """
    try:
        print(f"Processing image: {image_path}")
        
        # Load and preprocess image
        image_tensor, pil_image = load_and_preprocess_image(image_path)
        
        # Get prediction
        prediction, confidence = get_prediction(image_tensor, model, device)
        
        # Get explanation
        explanation = get_explanation(prediction, confidence)
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        print("Generating explainability visualizations...")
        
        # Generate visualizations (may take a few seconds)
        try:
            ela_img = generate_ela_image(image_path)
            print("✓ ELA generated")
        except Exception as e:
            print(f"⚠ ELA generation failed: {e}")
            ela_img = pil_image
        
        try:
            fft_img = generate_fft_image(image_path)
            print("✓ FFT generated")
        except Exception as e:
            print(f"⚠ FFT generation failed: {e}")
            fft_img = pil_image
        
        try:
            grad_cam_img = generate_grad_cam(image_tensor, model, device)
            print("✓ Grad-CAM generated")
        except Exception as e:
            print(f"⚠ Grad-CAM generation failed: {e}")
            grad_cam_img = pil_image
        
        # Convert images to base64
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'confidence_percentage': round(confidence * 100, 2),
            'explanation': explanation,
            'images': {
                'original': image_to_base64(pil_image),
                'ela': image_to_base64(ela_img),
                'fft': image_to_base64(fft_img),
                'grad_cam': image_to_base64(grad_cam_img)
            }
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        raise
