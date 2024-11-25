import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)
from torch.nn.functional import softmax
import threading
import time
from collections import deque

import sqlite3
from datetime import datetime

class MultiPersonVideoClassifier:
    def __init__(self, window, video_path):
        self.window = window
        self.window.title("Multi-Person Action Recognition")
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.yolo_model = YOLO("yolov8x-worldv2.pt")
        self.transform = Compose([
            Lambda(lambda x: x/255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        
        # Create display elements
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # FPS display
        self.fps_label = tk.Label(window, text="FPS: Calculating...", font=('Arial', 14), bg='white')
        self.fps_label.pack()
        
        # Video handling
        self.vid = cv2.VideoCapture(video_path)
        
        # Store frames for each person using deque for efficient frame management
        self.person_frames = {}  # Track ID -> deque of frames
        self.person_predictions = {}  # Track ID -> (action, confidence)
        
        # Add action duration tracking
        self.action_timestamps = {}  # Track ID -> {action: (start_time, current_duration)}
        self.current_actions = {}    # Track ID -> current_action
        
        # Class names
        self.class_names = ["Walking", "Standing", "Sitting", "Drinking", "Eating"]
        
        # Lock for thread-safe operations
        self.prediction_lock = threading.Lock()
        
        # Start video processing
        threading.Thread(target=self.update_video, daemon=True).start()
    
    def insertOrUpdate(self, track_id, action, duration, start_time):
        current_time = time.time()
        convert_start_time = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        convert_end_time = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        round_duration = f"{duration:.2f}"

        conn = sqlite3.connect('actionDB.db')
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS Person(
                    ID INTEGER PRIMARY KEY,
                    Name TEXT
                    )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS Action(
                    ActionID INTEGER PRIMARY KEY,
                    ActionName TEXT)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS PersonAction (
                    PersonID INTEGER,
                    ActionID INTEGER,
                    ActionName TEXT,
                    Duration REAL,
                    StartTime REAL,
                  EndTime Real,
                    FOREIGN KEY (PersonID) REFERENCES Person(ID),
                    FOREIGN KEY (ActionID) REFERENCES Action(ActionID)
                    )''')
        
        c.execute('SELECT * FROM Person WHERE ID = ?', (track_id,))
        if c.fetchone() is None:
            c.execute('INSERT INTO Person (ID, Name) VALUES (?,?)', (track_id, f'Person {track_id}'))

        # Ensure the action exists in the Action table
        c.execute('SELECT ActionID FROM Action WHERE ActionName = ?', (action,))
        action_row = c.fetchone()
        if action_row is None:
            c.execute('INSERT INTO Action (ActionName) VALUES (?)', (action,))
            action_id = c.lastrowid  # Get the newly inserted ActionID
        else:
            action_id = action_row[0]  # Use the existing ActionID

        c.execute('SELECT ActionID, Duration FROM PersonAction WHERE PersonID = ? ORDER BY rowid DESC LIMIT 1', (track_id,))
        last_action_row = c.fetchone()
        if last_action_row and last_action_row[0] == action_id:
            newDuration = round_duration
            c.execute('''UPDATE PersonAction 
                        SET Duration = ?, EndTime = ?
                        WHERE PersonID = ? AND ActionID = ? AND rowid = (
                        SELECT MAX(rowid) FROM PersonAction 
                        WHERE PersonID = ? AND ActionID = ?)''', (newDuration, convert_end_time, track_id, action_id, track_id, action_id))  
        else:
            # Insert the ActionID and PersonID log into the bridge PersonAction table
            c.execute('INSERT INTO PersonAction (PersonID, ActionID, ActionName, Duration, StartTime, EndTime) VALUES (?, ?, ?, ?, ?, ?)', 
                    (track_id, action_id, action, round_duration, convert_start_time, convert_end_time))

        # c.execute('UPDATE PersonAction SET Duration = ? WHERE PersonID = ? AND ActionName = ?', (duration, track_id, action))

        conn.commit()
        conn.close()


    def load_model(self):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 5)
        )
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.load_state_dict(torch.load("/home/sophic/Video_AI_Project/best_x3d_model(newupdate).pth", map_location=map_location))
        model.load_state_dict(torch.load("/home/sophic/Video_AI_Project/best_x3d_model(newupdate).pth"))
        model = model.to(self.device)
        model.eval()
        return model
    
    def update_action_duration(self, track_id, action):
        """Update action duration tracking"""
        current_time = time.time()
        
        # Initialize tracking for new person
        if track_id not in self.action_timestamps:
            self.action_timestamps[track_id] = {}
            self.current_actions[track_id] = None
        
        # If this is a new action for this person
        if self.current_actions[track_id] != action:
            # Store start time for new action
            self.action_timestamps[track_id][action] = (current_time, 0)
            self.current_actions[track_id] = action
        else:
            # Update duration for ongoing action
            start_time, _ = self.action_timestamps[track_id][action]
            duration = current_time - start_time
            self.action_timestamps[track_id][action] = (start_time, duration)
        
        return self.action_timestamps[track_id][action][1], self.action_timestamps[track_id][action][0]
    
    def extract_person_roi(self, frame, box):
        """Extract and resize ROI for a person"""
        x1, y1, x2, y2 = map(int, box)
        # Add padding while keeping within frame boundaries
        h, w = frame.shape[:2]
        pad = 100
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Resize to model input size while maintaining aspect ratio 480
        target_height = 182
        #target_width = int(w * (target_height / h))
        target_width = 182
        roi_resized = cv2.resize(roi, (target_width, target_height))
        return roi_resized

    def make_prediction(self, frames_list):
        """Make prediction for a sequence of frames"""
        if len(frames_list) != 4:
            return None, 0.0
        
        # Prepare frames for prediction
        frames_array = np.array(frames_list).astype(np.float32)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)
        frames_tensor = self.transform(frames_tensor).unsqueeze(0)
        frames_tensor = frames_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            # Top 1 Prediction
            probabilities = softmax(outputs, dim=1)[0]
            top_value, top_index = torch.max(probabilities, 0)
            
            return self.class_names[top_index], float(top_value)
        
    def get_video_duration(self):
        """Show Duration of the video"""
        try:
            fps = self.vid.get(cv2.CAP_PROP_FPS)
            total_frame = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frame/fps
            return duration, total_frame
        except Exception as e:
            raise ValueError(f"Error getting video duration: {e}")

    
    def update_video(self):
        last_time = time.time()
        frame_count = 0

        duration, total_frame = self.get_video_duration()
        #print(self.yolo_model)
        print(f"Video Duration: {duration:.2f} seconds")
        print(f"Total Frames: {total_frame} frames")
        
        
        while True:
            current_time = time.time()
            ret, frame = self.vid.read()
            
            if ret:
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                frame_count += 1
                
                print(f"\nFrame Now: {frame_count}")
                # Run YOLO detection
                results = self.yolo_model.track(frame_resized, persist=True, verbose=False)
                annotated_frame = frame_resized.copy()
                
                if results[0].boxes.id is not None:
                    # Filter for person class (class_id = 0)
                    mask = results[0].boxes.cls == 0
                    
                    if mask.any():
                        boxes = results[0].boxes.xyxy[mask]
                        track_ids = results[0].boxes.id[mask].cpu().numpy().astype(int)
                        print(f"See Track ID: {track_ids}")
                        
                        # Process each detected person
                        for box, track_id in zip(boxes, track_ids):
                            # Initialize frame deque for new tracks
                            if track_id not in self.person_frames:
                                self.person_frames[track_id] = deque(maxlen=4)
                            
                            # Extract and store ROI
                            roi = self.extract_person_roi(frame_resized, box)
                            if roi is not None:
                                self.person_frames[track_id].append(roi)
                                
                                if len(self.person_frames[track_id]) == 4:
                                    # Make prediction for this person
                                    frames_list = list(self.person_frames[track_id])
                                    action, confidence = self.make_prediction(frames_list)
                                    
                                    with self.prediction_lock:
                                        self.person_predictions[track_id] = (action, confidence)
                                        # Update duration for the predicted action
                                        duration, start_time = self.update_action_duration(track_id, action)
                                        self.person_predictions[track_id] = (action, confidence, duration, start_time)
                                    
                                    self.person_frames[track_id].clear()
                            
                            # Draw detection and prediction
                            self.draw_person_detection(annotated_frame, box, track_id)
                        
                        # Clean up old tracks
                        current_ids = set(track_ids)
                        stored_ids = set(self.person_frames.keys())
                        for old_id in stored_ids - current_ids:
                            self.person_frames.pop(old_id, None)
                            self.person_predictions.pop(old_id, None)
                            self.action_timestamps.pop(old_id, None)
                            self.current_actions.pop(old_id, None)
                
                # Display frame
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(annotated_frame))
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.photo = photo
                
                # Update FPS
                fps = 1/(time.time() - current_time)
                self.fps_label.config(text=f"FPS: {fps:.2f}")
                
                #time.sleep(1/30)  # Limit to approximately 30 FPS
            else:
                # print(f"Video Duration: {duration:.2f} seconds")
                break
    
    def draw_person_detection(self, frame, box, track_id):
        """Draw bounding box and prediction for a person"""
        x1, y1, x2, y2 = map(int, box)
        
        with self.prediction_lock:
            prediction = self.person_predictions.get(track_id)
        
        if prediction:
            action, confidence, duration, start_time = prediction
            # Color based on confidence (green intensity)
            color = (0, int(255 * confidence), 0)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"ID:{track_id} - {action} ({confidence:.2f}) - {duration:.1f} seconds"
            self.insertOrUpdate(int(track_id), action, duration, start_time)
            print(label)
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        else:
            # Draw box without prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{track_id}"
            print(label)
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

if __name__ == "__main__":
    video_path = "/home/sophic/Video_AI_Project/VideoPredict/sitStand(2).mp4"
    #video_path = "/home/sophic/Action-Recognition/archery.mp4"
    root = tk.Tk()
    root.configure(bg='white')
    app = MultiPersonVideoClassifier(root, video_path)
    root.mainloop()