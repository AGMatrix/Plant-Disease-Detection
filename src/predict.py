"""
Prediction module for DINOv2-based plant disease detection
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

from model import create_model


class DiseasePredictor:
    """Predictor using DINOv2 model"""
    
    def __init__(self, model_path='best_model.pth', classes_path='classes.json'):
        # Load classes
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load DINOv2 model
        print("Loading DINOv2 model...")
        self.model = create_model(len(self.classes), model_size='base', freeze_backbone=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"DINOv2 predictor ready on {self.device}")
    
    def predict(self, image_path, top_k=5):
        """Predict disease from image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.classes)))
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results.append({
                'disease': self.classes[idx],
                'confidence': float(prob * 100)
            })
        
        return results
