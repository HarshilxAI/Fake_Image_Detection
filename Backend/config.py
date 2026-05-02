"""
Backend Configuration for Fake Image Detector
"""
import os
import torch

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'checkpoint.pth')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # API settings
    JSON_SORT_KEYS = False
    PROPAGATE_EXCEPTIONS = True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Default config
config = DevelopmentConfig()
