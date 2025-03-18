import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

# Load dataset from the correct location
dataset = np.load(args.dataset)  # Shape: (121, 5, 8, 224, 224, 3)
subject_data = dataset[args.subject]  # Shape: (5, 8, 224, 224, 3)

# Load pretrained CNN (e.g., ResNet50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Transformation for images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Feature extraction
features = []
for region in range(5):  # Central, Frontal, Occipital, Parietal, Temporal
    for img_idx in range(8):  # 8 images per region
        img = subject_data[region, img_idx]  # Shape: (224, 224, 3)
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            feat = model(img)
        features.append(feat.cpu().numpy())

# Save extracted features
subject_features = np.array(features).reshape(5, 8, -1)  # Shape: (5, 8, feature_dim)
np.save(os.path.join(args.output_path, f"features_subject_{args.subject}.npy"), subject_features)
print(f"Features extracted for subject {args.subject}")
