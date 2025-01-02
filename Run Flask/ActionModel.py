import torch
from ultralytics import YOLO
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms._transforms_video import NormalizeVideo
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np

class ActionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = Compose([
            ShortSideScale(size=256), # Remove this for x3d-s
            Lambda(lambda x: x/255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        self.yolo_model = YOLO("yolov8x-worldv2.pt")
        self.class_names = ["Walking", "Standing", "Sitting", "Drinking", 
                           "Using Phone", "Using Laptop", "Talking", "Fall Down"]
    # first model x3d_xs
    def _load_model(self):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=False) # Load the x3d model, can change for x3d_s
        #in_features = model.blocks[5].proj.in_features   # Use this for x3d-s instead of below
        #model.blocks[5].proj = nn.Sequential(
         #   nn.Dropout(p=0.5),
         #   nn.Linear(in_features, 8)
        #)
        
        model.blocks[5].proj = nn.Linear(
          in_features=model.blocks[5].proj.in_features,
          out_features=8
        )   
        
        # first trained model best_x3d_model(NewDatasetMMAct9).pth
        model.load_state_dict(torch.load("best_x3d_model(NewDatasetMMActX3DM2).pth", 
                                       map_location=self.device))  # Load the weight of our trained model
        return model.to(self.device).eval()
    
    def predict(self, frames):
        
        print(f"Number of frame in predict {len(frames)}")
        
        frames_tensor = torch.from_numpy(np.array(frames).astype(np.float32))
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        
        print(f"tensor shape: {frames_tensor.shape}")
        frames_tensor = self.transform(frames_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = softmax(outputs, dim=1)[0] # Top 1 Prediction
            conf, idx = torch.max(probs, 0)    # Get confidence score and its action index
            return self.class_names[idx], float(conf)
