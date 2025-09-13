import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pickle
import os
import sys
import cv2

class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        return nn.functional.normalize(embedding, p=2, dim=1)

def enhance_fingerprint_image(img):
    """Enhance fingerprint image quality"""
    # Convert PIL to CV2 if needed
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    # Apply histogram equalization to improve contrast
    img_eq = cv2.equalizeHist(img_array)
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    
    return Image.fromarray(img_blur)

def create_multiple_templates(img_path, num_rotations=8):
    """Create multiple templates with different orientations"""
    img = Image.open(img_path).convert("L")
    
    # Enhance the original image
    img = enhance_fingerprint_image(img)
    
    templates = []
    
    # Original image
    templates.append(img)
    
    # Multiple rotations
    for angle in range(-15, 16, 30//num_rotations):  # -15 to +15 degrees
        if angle != 0:
            rotated = img.rotate(angle, expand=False, fillcolor=128)
            templates.append(rotated)
    
    # Slight scaling variations
    for scale in [0.95, 1.05]:
        size = img.size
        new_size = (int(size[0] * scale), int(size[1] * scale))
        scaled = img.resize(new_size, Image.Resampling.LANCZOS)
        # Crop or pad to original size
        if scale > 1:
            # Crop center
            left = (scaled.width - size[0]) // 2
            top = (scaled.height - size[1]) // 2
            scaled = scaled.crop((left, top, left + size[0], top + size[1]))
        else:
            # Pad to center
            new_img = Image.new('L', size, 128)
            left = (size[0] - scaled.width) // 2
            top = (size[1] - scaled.height) // 2
            new_img.paste(scaled, (left, top))
            scaled = new_img
        templates.append(scaled)
    
    return templates

def load_fingerprint_templates(path):
    """Load and preprocess multiple fingerprint templates"""
    templates = create_multiple_templates(path)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor_templates = []
    for template in templates:
        tensor_templates.append(transform(template).unsqueeze(0))
    
    return tensor_templates

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1.flatten(), vec2.flatten())

def enhanced_match_fingerprint(image_path, threshold=0.8, use_multiple_templates=True):
    """Enhanced matching with multiple templates and distortion handling"""
    print(f"🔍 Enhanced matching: {image_path}")
    
    # Check if database exists
    db_file = "fingerprint_database.pkl"
    if not os.path.exists(db_file):
        print("❌ No fingerprint database found!")
        return None
    
    # Load database
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    
    print(f"📊 Database contains {len(database)} fingerprints")
    
    # Initialize model
    torch.manual_seed(42)
    model = FingerprintEmbeddingNet(embedding_dim=512)
    model.eval()
    
    # Load probe fingerprint templates
    if use_multiple_templates:
        print("🔄 Creating multiple templates for probe image...")
        probe_templates = load_fingerprint_templates(image_path)
        print(f"✅ Created {len(probe_templates)} probe templates")
        
        # Compute embeddings for all probe templates
        probe_embeddings = []
        with torch.no_grad():
            for template in probe_templates:
                embedding = model(template).cpu().numpy()
                probe_embeddings.append(embedding)
    else:
        # Single template (original method)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = Image.open(image_path).convert("L")
        probe_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            probe_embeddings = [model(probe_tensor).cpu().numpy()]
    
    # Compare against database
    best_match = None
    best_score = -1
    all_matches = []
    
    print("\\n🎯 Comparing against database...")
    
    for person_name, person_data in database.items():
        db_embedding = person_data['embedding']
        
        if use_multiple_templates:
            # Find best match among all probe templates
            max_similarity = -1
            for probe_embedding in probe_embeddings:
                similarity = cosine_similarity(probe_embedding, db_embedding)
                max_similarity = max(max_similarity, similarity)
            similarity = max_similarity
        else:
            similarity = cosine_similarity(probe_embeddings[0], db_embedding)
        
        all_matches.append((person_name, similarity, person_data['image_path']))
        print(f"   {person_name}: {similarity:.3f}")
        
        if similarity > best_score:
            best_score = similarity
            best_match = person_name
    
    # Sort matches by similarity score
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\\n📈 Best match: {best_match} (score: {best_score:.3f})")
    print(f"🎚️  Threshold: {threshold}")
    
    # Decision based on threshold
    if best_score >= threshold:
        print(f"\\n✅ MATCH FOUND!")
        print(f"👤 Person: {best_match}")
        print(f"📊 Confidence: {best_score:.3f}")
        print(f"🖼️  Original image: {database[best_match]['image_path']}")
        return {
            'match': True,
            'person': best_match,
            'confidence': best_score,
            'all_matches': all_matches
        }
    else:
        print(f"\\n❌ NO MATCH FOUND")
        print(f"👤 Closest match: {best_match} (score: {best_score:.3f})")
        print("🔒 Score is below threshold")
        return {
            'match': False,
            'closest_person': best_match,
            'closest_score': best_score,
            'all_matches': all_matches
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python enhanced_match.py <image_path> [threshold] [--single-template]")
        print("Examples:")
        print("  python enhanced_match.py fingerprint3.bmp")
        print("  python enhanced_match.py fingerprint3.bmp 0.85")
        print("  python enhanced_match.py fingerprint3.bmp 0.8 --single-template")
        return
    
    image_path = sys.argv[1]
    threshold = 0.8
    use_multiple_templates = True
    
    # Parse arguments
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--single-template":
            use_multiple_templates = False
        else:
            try:
                threshold = float(arg)
            except ValueError:
                print(f"⚠️  Invalid argument: {arg}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return
    
    method_name = "Multiple Templates" if use_multiple_templates else "Single Template"
    print(f"🔧 Using: {method_name} Method")
    
    result = enhanced_match_fingerprint(image_path, threshold, use_multiple_templates)
    
    if result:
        print(f"\\n🏆 Top 3 matches:")
        for i, (person, score, image_path) in enumerate(result['all_matches'][:3]):
            print(f"   {i+1}. {person}: {score:.3f}")
        
        if result['match']:
            print(f"\\n🎉 Authentication SUCCESSFUL!")
        else:
            print(f"\\n🚫 Authentication FAILED!")

if __name__ == "__main__":
    main()