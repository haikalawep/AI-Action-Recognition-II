import torch
from ultralytics import YOLO
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np

class ActionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = Compose([
            Lambda(lambda x: x/255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        self.yolo_model = YOLO("yolov8x-worldv2.pt")
        self.class_names = ["Walking", "Standing", "Sitting", "Drinking", 
                           "Using Phone", "Using Laptop", "Talking", "Fall Down"]
    
    def _load_model(self):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 8)
        )
        model.load_state_dict(torch.load("best_x3d_model(NewDatasetMMAct9).pth", 
                                       map_location=self.device))
        return model.to(self.device).eval()
    
    # Make prediction for a each 4 frames
    def predict(self, frames):
        if len(frames) != 4:
            return None, 0.0
        
        frames_tensor = torch.from_numpy(np.array(frames).astype(np.float32))
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        frames_tensor = self.transform(frames_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():       
            outputs = self.model(frames_tensor)
            probabilities = softmax(outputs, dim=1)[0]       # Top 1 Prediction
            confidence, idx = torch.max(probabilities, 0)
            return self.class_names[idx], float(confidence)