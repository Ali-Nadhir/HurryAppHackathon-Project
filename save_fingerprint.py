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

class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Use ResNet18 backbone (pretrained on ImageNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

def save_fingerprint_to_db(image_path, person_name):
    """Save a fingerprint embedding to the database"""
    print(f"Processing fingerprint for: {person_name}")
    
    # Initialize model with fixed seed for consistency
    torch.manual_seed(42)  # Fixed seed for reproducible embeddings
    model = FingerprintEmbeddingNet(embedding_dim=512)
    model.eval()
    
    # Load and process the fingerprint image
    try:
        img_tensor = load_fingerprint(image_path)
        print("✅ Image loaded successfully")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Compute embedding
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    
    # Load existing database or create new one
    db_file = "fingerprint_database.pkl"
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            database = pickle.load(f)
        print(f"✅ Loaded existing database with {len(database)} fingerprints")
    else:
        database = {}
        print("✅ Created new database")
    
    # Save the fingerprint embedding
    database[person_name] = {
        'embedding': embedding,
        'image_path': image_path
    }
    
    # Save database back to file
    with open(db_file, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"✅ Fingerprint for '{person_name}' saved to database!")
    print(f"📊 Database now contains {len(database)} fingerprints")
    return True

def process_folder(folder_path):
    """Process all fingerprint images in a folder"""
    print(f"🗂️  Processing folder: {folder_path}")
    
    # Supported image extensions
    image_extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg', '*.BMP', '*.PNG', '*.JPG', '*.JPEG']
    
    # Find all image files in the folder
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(folder_path, extension)
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"❌ No supported image files found in '{folder_path}'")
        print("📁 Supported formats: BMP, PNG, JPG, JPEG")
        return False
    
    print(f"📸 Found {len(image_files)} image files")
    
    success_count = 0
    failed_count = 0
    
    for image_path in image_files:
        # Extract filename without extension as person name
        filename = os.path.basename(image_path)
        person_name = os.path.splitext(filename)[0]
        
        print(f"\n👤 Processing: {filename} → {person_name}")
        
        # Save fingerprint to database
        if save_fingerprint_to_db(image_path, person_name):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 Processing complete!")
    print(f"✅ Successfully processed: {success_count} fingerprints")
    if failed_count > 0:
        print(f"❌ Failed to process: {failed_count} fingerprints")
    
    return success_count > 0

def main():
    """Main function to save fingerprint(s)"""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage:")
        print("  Single file: python save_fingerprint.py <image_path> <person_name>")
        print("  Folder mode: python save_fingerprint.py <folder_path>")
        print("")
        print("Examples:")
        print("  python save_fingerprint.py finger1.bmp John_Doe")
        print("  python save_fingerprint.py ./fingerprint_images/")
        return
    
    path = sys.argv[1]
    
    # Check if path exists
    if not os.path.exists(path):
        print(f"❌ Error: Path '{path}' not found!")
        return
    
    # Check if it's a directory or file
    if os.path.isdir(path):
        # Folder mode - process all images in folder
        print("🗂️  Folder mode detected")
        success = process_folder(path)
        if success:
            print("\n🎉 Successfully processed folder!")
        else:
            print("\n❌ Failed to process folder")
    
    elif os.path.isfile(path):
        # Single file mode - need person name
        if len(sys.argv) != 3:
            print("❌ Error: Person name required for single file mode")
            print("Usage: python save_fingerprint.py <image_path> <person_name>")
            return
        
        person_name = sys.argv[2]
        print("📄 Single file mode detected")
        
        # Save fingerprint to database
        success = save_fingerprint_to_db(path, person_name)
        if success:
            print("\n🎉 Successfully saved fingerprint to database!")
        else:
            print("\n❌ Failed to save fingerprint")
    
    else:
        print(f"❌ Error: '{path}' is neither a file nor a directory!")

if __name__ == "__main__":
    main()