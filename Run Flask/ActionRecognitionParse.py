import argparse
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import threading
from collections import deque
import time
from ActionModel import ActionModel
from ActionDatabase import Database

class VideoProcessor:
    def __init__(self, window, video_path):
        self.window = window
        self.window.title("AI Human Action Recognition")
        # Create display elements
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        # FPS Display
        self.fps_label = tk.Label(window, text="FPS: Calculating...", font=('Arial', 14), bg='white')
        self.fps_label.pack()
        # Video handling
        self.vid = cv2.VideoCapture(video_path)
        self.action_model = ActionModel()
        self.db = Database()
        
        self.person_frames = {}         # Track ID -> deque of frames
        self.person_predictions = {}    # Track ID -> (action, confidence)
        self.action_timestamps = {}     # Track ID -> {action: (start_time, current_duration)}
        self.current_actions = {}       # Track ID -> current_action
        # Lock for thread-safe operations
        self.prediction_lock = threading.Lock()
        threading.Thread(target=self.process_video, daemon=True).start()    # Start video processing

    
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
        pad = 40
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
                results = self.action_model.yolo_model.track(frame_resized, persist=True, verbose=False, conf=0.2)
                print(self.action_model.device)
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
                                    action, confidence = self.action_model.predict(frames_list)
                                    
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw box
            label = f"ID:{track_id} - {action} ({confidence:.2f}) - {duration:.1f} seconds" # Draw label with background

            self.db.update_action(int(track_id), action, duration, start_time)  # Update detail into DB
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