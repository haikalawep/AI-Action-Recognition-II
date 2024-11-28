from flask import Flask, render_template, Response, redirect, url_for, request, flash, jsonify
import cv2
import torch
import numpy as np
import sqlite3
import time
import os
import threading

# Import YOLO and torch modules
from ultralytics import YOLO
import torch.nn as nn
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo
from torch.nn.functional import softmax
from datetime import datetime, timedelta
from collections import deque


app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Required for flash messages

class MultiPersonVideoClassifier:
    def __init__(self, video_path):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.yolo_model = YOLO("yolov8x-worldv2.pt")
        self.transform = Compose([
            Lambda(lambda x: x/255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])
        
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
        
        # Tracking video processing state
        self.is_processing = True
        self.current_frame = None

    def load_model(self):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
        in_features = model.blocks[5].proj.in_features
        model.blocks[5].proj = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 5)
        )
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load("best_x3d_model(newupdate).pth", map_location=map_location))
        model = model.to(self.device)
        model.eval()
        return model

    # [Keep all the other methods from the original class: 
    #  insertOrUpdate, extract_person_roi, make_prediction, 
    #  update_action_duration, get_video_duration, update_video, draw_person_detection]
    # (These methods remain exactly the same as in the original implementation)

    def insertOrUpdate(self, track_id, action, duration, start_time):
        current_time = time.time()
        # convert_start_time = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        # convert_end_time = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        # Start Time
        utc_time_start = datetime.utcfromtimestamp(start_time)
        malaysia_time_start = utc_time_start + timedelta(hours=8)
        convert_start_time = malaysia_time_start.strftime('%H:%M:%S')

        # End Time
        utc_time_end = datetime.utcfromtimestamp(current_time)
        malaysia_time_end = utc_time_end + timedelta(hours=8)
        convert_end_time = malaysia_time_end.strftime('%H:%M:%S')

        # convert_start_time = datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S')
        # convert_end_time = datetime.utcfromtimestamp(current_time).strftime('%H:%M:%S')
        round_duration = f"{duration:.2f}"
        
        DB_PATH = os.environ.get('SQLITE_DB_PATH', '/app/database/flaskDB.db')
        
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
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


    def process_video(self):
        """Process video frames and store results"""
        last_time = time.time()
        frame_count = 0

        duration, total_frame = self.get_video_duration()
        print(f"Video Duration: {duration:.2f} seconds")
        print(f"Total Frames: {total_frame} frames")
        
        while self.is_processing:
            ret, frame = self.vid.read()
            
            if ret:
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                frame_count += 1

                print(f"Frame Now: {frame_count}")
                
                # Run YOLO detection
                results = self.yolo_model.track(frame_resized, persist=True, verbose=False, conf=0.2)
                annotated_frame = frame_resized.copy()
                
                if results[0].boxes.id is not None:
                    # Filter for person class (class_id = 0)
                    mask = results[0].boxes.cls == 0
                    
                    if mask.any():
                        boxes = results[0].boxes.xyxy[mask]
                        track_ids = results[0].boxes.id[mask].cpu().numpy().astype(int)
                        
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
                                        # Update duration for the predicted action
                                        duration, start_time = self.update_action_duration(track_id, action)
                                        self.person_predictions[track_id] = (action, confidence, duration, start_time)
                                    
                                    self.person_frames[track_id].clear()
                            
                            # Draw detection and prediction
                            self.draw_person_detection(annotated_frame, box, track_id)
                
                # Update current frame
                self.current_frame = annotated_frame
            else:
                # End of video
                self.is_processing = False
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

    def get_frame(self):
        """Get the current processed frame for streaming"""
        if self.current_frame is not None:
            # Convert the frame color back to BGR
            frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            # Convert frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame_bgr)
            return jpeg.tobytes()
        return None

def generate_frames(classifier):
    """Generator function for streaming frames"""
    # Start video processing in a separate thread
    processing_thread = threading.Thread(target=classifier.process_video)
    processing_thread.start()

    while classifier.is_processing:
        frame = classifier.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)  # Control frame rate

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/model')
def open():
    return render_template('index.html')

@app.route('/video1')
def video():
    # For webcam
    classifier = MultiPersonVideoClassifier(0)  # 0 for webcam
    return Response(generate_frames(classifier), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videoResult', methods=['POST'])
def videoR():
    # Check if the request contains a file
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)

    # Get the uploaded video file
    video_file = request.files['video']

    # Specify the directory path to save the uploaded video
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the uploaded video to the specified directory
    video_path = os.path.join(upload_dir, 'temporary.mp4')
    video_file.save(video_path)

    # Pass the video path to the template
    return jsonify({'video_path': video_path})

@app.route('/displayVideo1/<video_name>')
def displayVideo1(video_name):
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    video_path = os.path.join(upload_dir, video_name)
    
    # Create classifier for the uploaded video
    classifier = MultiPersonVideoClassifier(video_path)
    return Response(generate_frames(classifier), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
