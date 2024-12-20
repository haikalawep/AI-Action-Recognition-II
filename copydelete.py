import argparse
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from torch.nn.functional import softmax
import threading
from collections import deque
import sqlite3
from datetime import datetime, timedelta
import time

class Database:
    def __init__(self, db_name='action-recognitionDB.db'):
        self.db_name = db_name
        self._init_tables()
    
    def _init_tables(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.executescript('''
                CREATE TABLE IF NOT EXISTS Person(
                    ID INTEGER PRIMARY KEY,
                    Name TEXT
                );
                CREATE TABLE IF NOT EXISTS Action(
                    ActionID INTEGER PRIMARY KEY,
                    ActionName TEXT
                );
                CREATE TABLE IF NOT EXISTS PersonAction(
                    PersonID INTEGER,
                    ActionID INTEGER,
                    ActionName TEXT,
                    Duration REAL,
                    StartTime REAL,
                    EndTime REAL,
                    FOREIGN KEY (PersonID) REFERENCES Person(ID),
                    FOREIGN KEY (ActionID) REFERENCES Action(ActionID)
                );
            ''')
    
    def update_action(self, track_id, action, duration, start_time):
        current_time = time.time()
        start_time_malaysia = datetime.utcfromtimestamp(start_time) + timedelta(hours=8)
        end_time_malaysia = datetime.utcfromtimestamp(current_time) + timedelta(hours=8)
        
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            
            # Ensure person exists
            c.execute('INSERT OR IGNORE INTO Person (ID, Name) VALUES (?, ?)', 
                     (track_id, f'Person {track_id}'))
            
            # Ensure action exists
            c.execute('INSERT OR IGNORE INTO Action (ActionName) VALUES (?)', (action,))
            c.execute('SELECT ActionID FROM Action WHERE ActionName = ?', (action,))
            action_id = c.fetchone()[0]
            
            # Update or insert action record
            c.execute('''
                INSERT INTO PersonAction (PersonID, ActionID, ActionName, Duration, StartTime, EndTime)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (track_id, action_id, action, round(duration, 2),
                  start_time_malaysia.strftime('%H:%M:%S'),
                  end_time_malaysia.strftime('%H:%M:%S')))

class ActionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transforms()
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
    
    def _get_transforms(self):
        return Compose([
            Lambda(lambda x: x/255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
    
    def predict(self, frames):
        if len(frames) != 4:
            return None, 0.0
        
        frames_tensor = torch.from_numpy(np.array(frames).astype(np.float32))
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        frames_tensor = self.transform(frames_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            return self.class_names[idx], float(conf)

class VideoProcessor:
    def __init__(self, window, video_path):
        self.window = window
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.fps_label = tk.Label(window, text="FPS: Calculating...", 
                                font=('Arial', 14), bg='white')
        self.canvas.pack()
        self.fps_label.pack()
        
        self.vid = cv2.VideoCapture(video_path)
        self.action_model = ActionModel()
        self.db = Database()
        
        self.person_frames = {}
        self.person_predictions = {}
        self.action_timestamps = {}
        self.current_actions = {}
        
        self.prediction_lock = threading.Lock()
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        results = self.action_model.yolo_model.track(frame_resized, persist=True, 
                                                   verbose=False, conf=0.2)
        
        if results[0].boxes.id is not None:
            mask = results[0].boxes.cls == 0
            if mask.any():
                self._process_detections(frame_resized, results[0], mask)
        
        return frame_resized
    
    def _process_detections(self, frame, results, mask):
        boxes = results.boxes.xyxy[mask]
        track_ids = results.boxes.id[mask].cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, track_ids):
            if track_id not in self.person_frames:
                self.person_frames[track_id] = deque(maxlen=4)
            
            roi = self._extract_roi(frame, box)
            if roi is not None:
                self._update_predictions(track_id, roi, box, frame)
    
    def _extract_roi(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        pad = 40
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        return cv2.resize(roi, (182, 182)) if roi.size > 0 else None
    
    def _update_predictions(self, track_id, roi, box, frame):
        self.person_frames[track_id].append(roi)
        
        if len(self.person_frames[track_id]) == 4:
            action, conf = self.action_model.predict(list(self.person_frames[track_id]))
            
            if action:
                duration, start_time = self._update_action_duration(track_id, action)
                with self.prediction_lock:
                    self.person_predictions[track_id] = (action, conf, duration, start_time)
                    self.db.update_action(track_id, action, duration, start_time)
            
            self.person_frames[track_id].clear()
    
    def _update_action_duration(self, track_id, action):
        current_time = time.time()
        
        if track_id not in self.action_timestamps:
            self.action_timestamps[track_id] = {}
            self.current_actions[track_id] = None
        
        if self.current_actions[track_id] != action:
            self.action_timestamps[track_id][action] = (current_time, 0)
            self.current_actions[track_id] = action
        else:
            start_time, _ = self.action_timestamps[track_id][action]
            duration = current_time - start_time
            self.action_timestamps[track_id][action] = (start_time, duration)
        
        return self.action_timestamps[track_id][action]
    
    def process_video(self):
        while True:
            start_time = time.time()
            ret, frame = self.vid.read()
            
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            self._update_display(processed_frame, start_time)
    
    def _update_display(self, frame, start_time):
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.photo = photo
        
        fps = 1/(time.time() - start_time)
        self.fps_label.config(text=f"FPS: {fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Person Action Recognition')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    args = parser.parse_args()
    
    try:
        with open(args.video_path, 'r'):
            pass
    except IOError:
        print(f"Error: Video file '{args.video_path}' does not exist or is not readable.")
        return
    
    root = tk.Tk()
    root.configure(bg='white')
    VideoProcessor(root, args.video_path)
    root.mainloop()

if __name__ == "__main__":
    main()