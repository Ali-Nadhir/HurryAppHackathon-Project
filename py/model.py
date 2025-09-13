import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import glob

class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        # Use ResNet18 backbone (pretrained on ImageNet)
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # Add a fully-connected layer to produce fixed-size embedding
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        # x: (B, 3, 256, 288) fingerprint image (normalized)
        features = self.feature_extractor(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        embedding = self.embedding(features)  # (B, embedding_dim)
        # Normalize to unit length for cosine similarity
        return nn.functional.normalize(embedding, p=2, dim=1)

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # fingerprints are gray
    transforms.Resize((256, 288)),                 # standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])      # normalize to [-1, 1]
])

def load_fingerprint(path):
    """Load and preprocess fingerprint image"""
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)  # (1, 3, 256, 288)

def get_fingerprint_embedding(image_path, person_name):
    """Save a fingerprint embedding to the database"""
    print(f"Processing fingerprint for: {person_name}")
    
    # Initialize model with fixed seed for consistency
    torch.manual_seed(42)  # Fixed seed for reproducible embeddings
    model = FingerprintEmbeddingNet(embedding_dim=2048)
    model.eval()
    
    # Load and process the fingerprint image
    try:
        img_tensor = load_fingerprint(image_path)
        print("Image loaded successfully")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Compute embedding
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
        return embedding
    # Load existing database or create new one
