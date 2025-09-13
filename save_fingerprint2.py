import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import os
import sys
import glob

# -------------------------------
# Fingerprint embedding model
# -------------------------------
class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=2048):
        super().__init__()
        # Use ResNet152 backbone (pretrained on ImageNet)
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # Add fully-connected layer to produce fixed-size embedding
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)        # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        embedding = self.embedding(features)       # (B, embedding_dim)
        return nn.functional.normalize(embedding, p=2, dim=1)  # unit vector

# -------------------------------
# Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # fingerprints are gray
    transforms.Resize((256, 288)),                 # standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])      # normalize to [-1, 1]
])

def load_fingerprint(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)  # (1, 3, 256, 288)

# -------------------------------
# Save fingerprint embedding to database
# -------------------------------
def save_fingerprint_to_db(image_path, person_name):
    print(f"Processing fingerprint for: {person_name}")
    
    torch.manual_seed(42)
    model = FingerprintEmbeddingNet(embedding_dim=2048)
    model.eval()
    
    try:
        img_tensor = load_fingerprint(image_path)
        print("✅ Image loaded successfully")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    
    db_file = "fingerprint_database.pkl"
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            database = pickle.load(f)
        print(f"✅ Loaded existing database with {len(database)} fingerprints")
    else:
        database = {}
        print("✅ Created new database")
    
    database[person_name] = {
        'embedding': embedding,
        'image_path': image_path
    }
    
    with open(db_file, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"✅ Fingerprint for '{person_name}' saved to database!")
    print(f"📊 Database now contains {len(database)} fingerprints")
    return True

# -------------------------------
# Process folder of fingerprints
# -------------------------------
def process_folder(folder_path):
    print(f"🗂️  Processing folder: {folder_path}")
    
    image_extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.BMP', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"❌ No supported image files found in '{folder_path}'")
        return False
    
    print(f"📸 Found {len(image_files)} image files")
    
    success_count = 0
    failed_count = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        person_name = os.path.splitext(filename)[0]
        print(f"\n👤 Processing: {filename} → {person_name}")
        
        if save_fingerprint_to_db(image_path, person_name):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Processing complete! ✅ {success_count} / ❌ {failed_count}")
    return success_count > 0

# -------------------------------
# Main function
# -------------------------------
def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage:")
        print("  Single file: python save_fingerprint.py <image_path> <person_name>")
        print("  Folder mode: python save_fingerprint.py <folder_path>")
        return
    
    path = sys.argv[1]
    
    if not os.path.exists(path):
        print(f"❌ Error: Path '{path}' not found!")
        return
    
    if os.path.isdir(path):
        print("🗂️  Folder mode detected")
        process_folder(path)
    elif os.path.isfile(path):
        if len(sys.argv) != 3:
            print("❌ Person name required for single file mode")
            return
        person_name = sys.argv[2]
        print("📄 Single file mode detected")
        save_fingerprint_to_db(path, person_name)
    else:
        print(f"❌ Error: '{path}' is neither a file nor a directory!")

if __name__ == "__main__":
    main()
