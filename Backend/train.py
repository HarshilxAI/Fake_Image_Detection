"""
Model training script for ResNet50 on CIFAKE dataset
Trains a ResNet50 model to classify real vs AI-generated images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms, datasets
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

# Configuration
CONFIG = {
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'model_path': 'backend/models/checkpoint.pth',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42
}

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model(num_classes=2):
    """Create ResNet50 model for binary classification"""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def create_dummy_dataset(num_real=100, num_fake=100):
    """
    Create dummy dataset for testing
    In production, use actual CIFAKE dataset
    """
    print(f"Creating dummy dataset: {num_real} real + {num_fake} fake images")
    
    # Generate dummy images (random tensors)
    real_images = torch.randn(num_real, 3, 224, 224)
    fake_images = torch.randn(num_fake, 3, 224, 224)
    
    # Create labels (0 = Real, 1 = AI-generated)
    real_labels = torch.zeros(num_real, dtype=torch.long)
    fake_labels = torch.ones(num_fake, dtype=torch.long)
    
    # Combine
    images = torch.cat([real_images, fake_images], dim=0)
    labels = torch.cat([real_labels, fake_labels], dim=0)
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    
    # Split into train/val
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def get_data_loaders(train_dataset, val_dataset, batch_size):
    """Create data loaders"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, loss, acc, save_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")

def train():
    """Main training function"""
    print("\n" + "="*60)
    print("ResNet50 Fine-tuning for AI Image Detection")
    print("="*60 + "\n")
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Get device
    device = CONFIG['device']
    print(f"Using device: {device}\n")
    
    # Create model
    print("Creating ResNet50 model...")
    model = create_model(num_classes=2)
    model.to(device)
    print(f"✓ Model created\n")
    
    # Create dummy dataset (replace with real CIFAKE dataset)
    print("Loading dataset...")
    train_dataset, val_dataset = create_dummy_dataset(num_real=500, num_fake=500)
    print(f"✓ Dataset loaded\n")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        train_dataset, val_dataset, CONFIG['batch_size']
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("="*60)
    print("Training Configuration:")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Optimizer: Adam")
    print("="*60 + "\n")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print results
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, CONFIG['model_path'])
        
        # Learning rate scheduling
        scheduler.step()
    
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CONFIG['model_path']}")
    print("="*60 + "\n")

if __name__ == '__main__':
    train()
