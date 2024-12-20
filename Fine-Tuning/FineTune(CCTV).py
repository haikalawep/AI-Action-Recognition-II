# IMPORT ALL PACKAGE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from ultralytics import YOLO

from pytorchvideo.transforms import Normalize
from torchvision.transforms import Compose

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Custom Dataset Class
class CustomActionDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=4, padding=50):
        """
        video_paths: List of paths to video files
        labels: List of corresponding labels
        transform: Optional transform to be applied on video
        num_frames: Number of frames to sample from each video
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.padding = padding 
        self.yolo_model = YOLO("yolov8x-worldv2.pt")

    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)  # Get number of frame per video [Exp: 100 frames:- 0, 25, 50, 75]
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in indices:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame, (640, 640))

                # Apply person detection
                boxes, labels, scores = self.detect_person(frame_resized)
                
                if boxes:
                    # Apply padding around the bounding box
                    x1, y1, x2, y2 = boxes[0]  # Taking the first detected person
                    x1 = max(0, x1 - self.padding)
                    y1 = max(0, y1 - self.padding)
                    x2 = min(frame_rgb.shape[1], x2 + self.padding)
                    y2 = min(frame_rgb.shape[0], y2 + self.padding)

                    # Crop the frame to the adjusted bounding box
                    frame_rgb = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                
                # Resize the cropped frame
                frame_rgb = cv2.resize(frame_rgb, (182, 182))  # Resize for model input
                frames.append(frame_rgb)
                
        cap.release()
        return np.array(frames)
    

    def detect_person(self, frame):
        # Convert frame to tensor
        image = torch.tensor(frame).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        # Detect persons in the frame
        with torch.no_grad():
            prediction = self.yolo_model(image, verbose=False)
        
        # Get boxes with label=1 (person) and confidence score > 0.5
        boxes = prediction[0].boxes.xyxy.cpu().numpy()
        scores = prediction[0].boxes.conf.cpu().numpy()
        labels = prediction[0].boxes.cls.cpu().numpy()
        
        # Filter boxes for label=1 (person) and score > 0.5
        person_boxes = [boxes[i] for i in range(len(labels)) if labels[i] == 0 and scores[i] > 0.5]  # Assuming 0 = 'person

        return person_boxes, labels, scores


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.load_video(video_path)
        
        # Convert to float32 and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        video = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        if self.transform:
            video = self.transform(video)
            
        return video, label


# Function to collect dataset
def collect_dataset(root_dir):
    video_paths = []
    labels = []
    class_to_idx = {'Walking': 0, 'Standing': 1, 'Sitting': 2, 'Drinking': 3, 'Using Phone': 4, 'Using Laptop': 5, 'Talking': 6, 'Fall Down': 7}    # Set Label Action with Index
    
    for action in class_to_idx.keys():
        action_dir = os.path.join(root_dir, action)
        # Skip the next action if directory not exist
        if not os.path.exists(action_dir):
            continue
            
        for video_file in os.listdir(action_dir):
            if video_file.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(action_dir, video_file))
                labels.append(class_to_idx[action])
                
    return video_paths, labels



# Modify X3D head
def modify_x3d_head(model, num_classes=8):
    """Modify the classification head of X3D model with dropout and freeze/unfreeze layers"""
    # 1. Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    
    # 4. Unfreeze just the modified head (block 4 & 5)
    for param in model.blocks[4].parameters():
        param.requires_grad = True
    for param in model.blocks[5].parameters():
        param.requires_grad = True

    # 2. Get input features from the last block
    in_features = model.blocks[5].proj.in_features
    
    # 3. Modify the head with new number of classes
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.blocks[5].activation = None
        
    return model


# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for videos, labels in tqdm(train_loader, desc='Training'):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(val_loader, desc='Validation'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc



# Plot training history
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Evaluation function for confusion matrix and classification report
def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate the model and generate confusion matrix and classification report
    
    Args:
    - model: Trained PyTorch model
    - test_loader: DataLoader for test dataset
    - device: torch device (cuda/cpu)
    - class_names: List of class names
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc='Evaluating'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def main():
    # Configuration
    DATA_ROOT = "/home/sophic/Video_AI_Project/Dataset/Human Activity Recognition - Video Dataset/" # Update this path
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    model_name = 'x3d_xs'
    transform_params = {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
    }
    
    # Collect dataset
    video_paths, labels = collect_dataset(DATA_ROOT)
    
    # Split dataset
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, train_size=TRAIN_SIZE, stratify=labels, random_state=42
    )
    
    # Further split temp into validation and test
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, train_size=val_ratio, stratify=temp_labels, random_state=42
    )
    
    print(f"Train size: {len(train_paths)}")
    print(f"Validation size: {len(val_paths)}")
    print(f"Test size: {len(test_paths)}")
    
    # Define transforms
    train_transform = Compose([
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])
    
    val_transform = Compose([
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])
    
    
    # Create datasets
    train_dataset = CustomActionDataset(
        train_paths, train_labels,
        transform=train_transform,
        num_frames=transform_params["num_frames"]
    )
    
    val_dataset = CustomActionDataset(
        val_paths, val_labels,
        transform=val_transform,
        num_frames=transform_params["num_frames"]
    )
    
    test_dataset = CustomActionDataset(
        test_paths, test_labels,
        transform=val_transform,
        num_frames=transform_params["num_frames"]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load and modify model
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    print(model)
    model = modify_x3d_head(model, num_classes=8)
    print("Modify Model")
    print(model)
    model = model.to(device)
    
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=NUM_EPOCHS, 
    eta_min=1e-5
)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_x3d_model(NewDatasetMMAct8).pth')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate on test set
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    # Define class names to match your class_to_idx in collect_dataset
    class_names = ['Walking', 'Standing', 'Sitting', 'Drinking', 'Using Phone', 'Using Laptop', 'Talking', 'Fall Down']
    
    # Evaluate model with confusion matrix and classification report
    evaluate_model(model, test_loader, device, class_names)

# Run training
if __name__ == "__main__":
    main()