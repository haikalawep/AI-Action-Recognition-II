import os
from ultralytics import YOLO
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

# Define keypoint labels
KEYPOINT_LABELS = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE'
}

def get_frame_indices(total_frames, num_samples=16):
    """
    Calculate indices for equally spaced frames | Example: 64 frames ---> 1,4,8,12,16...,64
    """
    return np.linspace(0, total_frames - 1, num_samples, dtype=int)

def extract_keypoints(video_path, model, num_frames=16):
    """
    Extract keypoints from exactly 16 equally spaced frames in a video
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frame indices we want to sample
    frame_indices = get_frame_indices(total_frames, num_frames)
    keypoints_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            # Save frame to temporary file (YOLO expects a file path)
            temp_frame_path = 'temp_frame.jpg'
            cv2.imwrite(temp_frame_path, frame)
            
            # Process frame with YOLO
            results = model(source=temp_frame_path, show=False, conf=0.3, save=False)
            
            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                # Get keypoints for the first person detected
                keypoints = results[0].keypoints.xyn.cpu().numpy()[0]
                keypoints_flat = keypoints.flatten()
                keypoints_list.append(keypoints_flat)
            else:
                # If no keypoints detected, add zeros
                print("Not detect")
                keypoints_list.append(np.zeros(34))
                break
                
            # Clean up temporary file
            os.remove(temp_frame_path)
            
        frame_count += 1
        
    cap.release()
    
    # Ensure we have exactly num_frames frames
    if len(keypoints_list) < num_frames:
        # Pad with zeros if we couldn't get enough frames
        padding = num_frames - len(keypoints_list)
        keypoints_list.extend([np.zeros(34)] * padding)
    
    return np.array(keypoints_list)

def process_videos(base_dir, model, action):
    """
    Process all videos for a specific action and create a dataset
    """
    action_dir = os.path.join(base_dir, action)
    if not os.path.exists(action_dir):
        print(f"Directory not found: {action_dir}")
        return None
        
    data = []
    video_files = [f for f in os.listdir(action_dir) 
                  if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Processing {action} videos...")
    for video_file in tqdm(video_files):
        video_path = os.path.join(action_dir, video_file)
        
        try:
            # Extract frames of keypoints
            keypoints_frames = extract_keypoints(video_path, model)
            
            # Create rows for each frame
            for frame_idx, keypoints in enumerate(keypoints_frames):
                row = {
                    'video_name': video_file,
                    'frame_number': frame_idx,
                    'action': action
                }
                # Add keypoint coordinates with proper labels
                for i in range(len(KEYPOINT_LABELS)):
                    # Each keypoint has x and y coordinates
                    x_idx = i * 2
                    y_idx = i * 2 + 1
                    joint_name = KEYPOINT_LABELS[i]
                    row[f'{joint_name}_x'] = keypoints[x_idx]
                    row[f'{joint_name}_y'] = keypoints[y_idx]
                
                data.append(row)
                
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
    
    return pd.DataFrame(data)

def main():
    # Initialize YOLO model
    model = YOLO('yolov8m-pose.pt')
    
    # Set base directory containing action folders
    base_dir = "/home/sophic/Video_AI_Project/Dataset/Human Activity Recognition - Video Dataset/"
    
    # Process each action separately | Can add more action
    action_classes = ['Walking', 'Standing', 'Sitting']
    
    for action in action_classes:
        # Process videos for this action
        df = process_videos(base_dir, model, action)
        
        if df is not None:
            # Save to CSV with actions name
            output_file = f"{action.lower()}.csv"
            df.to_csv(output_file, index=False)
            print(f"Dataset for {action} saved to {output_file}")
            print(f"Total frames: {len(df)}")
            print(f"Number of videos: {len(df['video_name'].unique())}")
            print("-------------------")

if __name__ == "__main__":
    main()