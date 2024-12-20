from flask import Flask, render_template, Response, redirect, request, flash, jsonify
import cv2
import time
import os
import threading
from ActionModel import ActionModel
from ActionDatabase import Database
from collections import deque

app = Flask(__name__)

class FlaskActionClassifier:
    def __init__(self, video_path):
        # Video handling
        self.vid = cv2.VideoCapture(video_path)
        self.action_model = ActionModel()
        self.db = Database()
        
        # Store frames for each person using deque for efficient frame management
        self.person_frames = {}  # Track ID -> deque of frames
        self.person_predictions = {}  # Track ID -> (action, confidence)  
        # Add action duration tracking
        self.action_timestamps = {}  # Track ID -> {action: (start_time, current_duration)}
        self.current_actions = {}    # Track ID -> current_action
        
        # Class names
        self.class_names = ["Walking", "Standing", "Sitting", "Drinking", "Using Phone", "Using Laptop", "Talking", "Fall Down"]
        
        # Lock for thread-safe operations
        self.prediction_lock = threading.Lock()
        
        # Tracking video processing state
        self.is_processing = True
        self.current_frame = None


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
        """Extract and resize Region of Interest (ROI) for a person"""

        x1, y1, x2, y2 = map(int, box) # Get Bounding Box Coordinates

        h, w = frame.shape[:2]  # Add padding while keeping within frame boundaries
        pad = 50
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2] # New ROI after Expanding using Padding
        if roi.size == 0:
            return None
            
        # Resize to Maintain Aspect Ration 182 for X3D_XS
        target_height = 182
        target_width = 182
        roi_resized = cv2.resize(roi, (target_width, target_height))
        return roi_resized
        

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
                results = self.action_model.yolo_model.track(frame_resized, persist=True, verbose=False, conf=0.2)
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
                                    action, confidence = self.action_model.predict(frames_list)
                                    
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
            
            color = (0, int(255 * confidence), 0) # Color based on confidence (green intensity)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"ID:{track_id} - {action} ({confidence:.2f}) - {duration:.1f} seconds"
            self.db.update_action(int(track_id), action, duration, start_time)
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
    classifier = FlaskActionClassifier(0)  # 0 for webcam
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
    classifier = FlaskActionClassifier(video_path)
    return Response(generate_frames(classifier), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
