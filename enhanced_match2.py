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

# -------------------------------
# Fingerprint embedding model
# -------------------------------
class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        return nn.functional.normalize(embedding, p=2, dim=1)

# -------------------------------
# Fingerprint image enhancement
# -------------------------------
def enhance_fingerprint_image(img):
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img
    
    img_eq = cv2.equalizeHist(img_array)
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    return Image.fromarray(img_blur)

# -------------------------------
# Create multiple templates
# -------------------------------
def create_multiple_templates(img_path, num_rotations=8):
    img = Image.open(img_path).convert("L")
    img = enhance_fingerprint_image(img)
    
    templates = [img]  # original image
    
    for angle in range(-15, 16, 30 // num_rotations):
        if angle != 0:
            rotated = img.rotate(angle, expand=False, fillcolor=128)
            templates.append(rotated)
    
    for scale in [0.95, 1.05]:
        size = img.size
        new_size = (int(size[0] * scale), int(size[1] * scale))
        scaled = img.resize(new_size, Image.Resampling.LANCZOS)
        
        if scale > 1:
            left = (scaled.width - size[0]) // 2
            top = (scaled.height - size[1]) // 2
            scaled = scaled.crop((left, top, left + size[0], top + size[1]))
        else:
            new_img = Image.new('L', size, 128)
            left = (size[0] - scaled.width) // 2
            top = (size[1] - scaled.height) // 2
            new_img.paste(scaled, (left, top))
            scaled = new_img
        
        templates.append(scaled)
    
    return templates

# -------------------------------
# Load fingerprint templates
# -------------------------------
def load_fingerprint_templates(path):
    templates = create_multiple_templates(path)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor_templates = [transform(t).unsqueeze(0) for t in templates]
    return tensor_templates

# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1.flatten(), vec2.flatten())

# -------------------------------
# Enhanced matching
# -------------------------------
def enhanced_match_fingerprint(image_path, threshold=0.8, use_multiple_templates=True):
    print(f"🔍 Matching: {image_path}")
    
    db_file = "fingerprint_database.pkl"
    if not os.path.exists(db_file):
        print("❌ No fingerprint database found!")
        return None
    
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    
    print(f"📊 Database contains {len(database)} fingerprints")
    
    torch.manual_seed(42)
    model = FingerprintEmbeddingNet(embedding_dim=2048)
    model.eval()
    
    if use_multiple_templates:
        print("🔄 Creating multiple templates...")
        probe_templates = load_fingerprint_templates(image_path)
        probe_embeddings = []
        with torch.no_grad():
            for template in probe_templates:
                probe_embeddings.append(model(template).cpu().numpy())
    else:
        img = Image.open(image_path).convert("L")
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        probe_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probe_embeddings = [model(probe_tensor).cpu().numpy()]
    
    best_match = None
    best_score = -1
    all_matches = []
    
    for person_name, person_data in database.items():
        db_embedding = person_data['embedding']
        
        if use_multiple_templates:
            max_similarity = max(cosine_similarity(pe, db_embedding) for pe in probe_embeddings)
            similarity = max_similarity
        else:
            similarity = cosine_similarity(probe_embeddings[0], db_embedding)
        
        all_matches.append((person_name, similarity, person_data['image_path']))
        if similarity > best_score:
            best_score = similarity
            best_match = person_name
    
    all_matches.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 Best match: {best_match} (score: {best_score:.3f})")
    
    if best_score >= threshold:
        return {
            'match': True,
            'person': best_match,
            'confidence': best_score,
            'all_matches': all_matches
        }
    else:
        return {
            'match': False,
            'closest_person': best_match,
            'closest_score': best_score,
            'all_matches': all_matches
        }

# -------------------------------
# Main function
# -------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python match_fingerprint.py <image_path> [threshold] [--single-template]")
        return
    
    image_path = sys.argv[1]
    threshold = 0.8
    use_multiple_templates = True
    
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "--single-template":
            use_multiple_templates = False
        else:
            try:
                threshold = float(arg)
            except ValueError:
                print(f"⚠️ Invalid argument: {arg}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file '{image_path}' not found!")
        return
    
    result = enhanced_match_fingerprint(image_path, threshold, use_multiple_templates)
    
    if result:
        print(f"\n🏆 Top 3 matches:")
        for i, (person, score, path) in enumerate(result['all_matches'][:3]):
            print(f"   {i+1}. {person}: {score:.3f}")
        if result['match']:
            print("\n🎉 Authentication SUCCESSFUL!")
        else:
            print("\n🚫 Authentication FAILED!")

if __name__ == "__main__":
    main()
