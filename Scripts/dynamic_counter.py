import os
import json
import cv2
import time
import csv
import random
from datetime import timedelta
# import numpy as np
import pandas as pd
from ultralytics import YOLO
# from datetime import datetime
from ultralytics.utils.plotting import Annotator, colors

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap

def get_video_properties(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)))

def load_mask(mask_path, height, width):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != (height, width):
        raise ValueError("Mask image size must match video frame size.")
    return mask

def save_thumbnail(track_id, frame, output_folder, count, direction):
    """ Saves a frame as a thumbnail and logs the event in a CSV file. """
    os.makedirs(output_folder, exist_ok=True)
    thumbnail_path = os.path.join(output_folder, f"thumbnail_{count}.jpg")
    cv2.imwrite(thumbnail_path, frame)

    # Log to CSV
    csv_path = os.path.join(output_folder, "footfall_log.csv")
    log_footfall(track_id, csv_path, count, direction, thumbnail_path)

    print(f"Saved thumbnail: {thumbnail_path}")

def log_footfall(track_id, csv_path, count, direction, thumbnail_path):
    """ Logs movement detection in a CSV file. """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["track_id","Timestamp", "Movement_Direction", "Thumbnail_Path"])

        writer.writerow([track_id, time.strftime("%Y-%m-%d %H:%M:%S"), direction, thumbnail_path])

def get_color_for_id(person_id):
    random.seed(person_id)  # Seed the random generator with the ID
    r = random.randint(50, 255)  # Avoid very dark colors
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return (b, g, r)

def process_video(video_path, mask_path, model_path, output_folder, fps_skip=5 ):
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Error: Mask not loaded. Check the mask file path.")
        return

    # Load YOLO model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Video FPS
    frame_interval = max(1, frame_rate //fps_skip)  # Process frames at specified FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup Video Writer (Save Processed Video)
    output_video_path = os.path.join(output_folder, "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate , (frame_width, frame_height))

    video_start_time = time.time()  # Track when processing started

    last_positions = {}
    tracking_data = []
    frame_count = 0  
    in_counter = 0
    out_counter = 0

    # CSV output setup
    csv_output_path = os.path.join(output_folder, "tracking_results.csv")
    os.makedirs(output_folder, exist_ok=True)

    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue  # Skip frames to process every `fps_skip` FPS

        # Calculate timestamp
        elapsed_time = time.time() - video_start_time
        timestamp = str(timedelta(seconds=int(elapsed_time)))

        # fps = cap.get(cv2.CAP_PROP_FPS)  # Get video frame rate
        # frame_time_seconds = frame_count / fps  # Convert frame count to time
        # timestamp = str(datetime.timedelta(seconds=int(frame_time_seconds)))  # Convert to HH:MM:SS

        h, w, _ = frame.shape

        # YOLO Object Detection & Tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            track_ids = results[0].boxes.id  # Track IDs

            if track_ids is not None:
                track_ids = track_ids.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)  # Assign None if no tracking ID

            bottom_midpoint_y, bottom_midpoint_x = 0, 0
            counted_ids_in = set()
            counted_ids_out = set()


            for box, track_id in zip(boxes, track_ids):
                if track_id is None:
                    continue  # Skip if no valid tracking ID

                x1, y1, x2, y2 = map(int, box)
                bottom_midpoint_x = (x1 + x2) // 2
                bottom_midpoint_y = y2  # Bottom center of the box

                # Debugging logs for ID tracking
                print(f"Track ID: {track_id}, Position: ({bottom_midpoint_x}, {bottom_midpoint_y})")
                
                movement_direction = "None"
                bottom_midpoint_y = min(bottom_midpoint_y, mask.shape[0] - 1)
                bottom_midpoint_x = min(bottom_midpoint_x, mask.shape[1] - 1)

                # Draw bounding box and track ID
                color = get_color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

                if track_id in last_positions:
                    last_x, last_y, last_side = last_positions[track_id]
                    current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]
                    current_side = 255 if current_pixel == 255 else 0  # Define side

                    # Check movement direction
                    if last_side == 0 and current_side == 255 and (last_y - bottom_midpoint_y > 4):  # Ensure significant movement
                        if track_id not in counted_ids_in:  # Prevent duplicate "IN" counts
                            movement_direction = "In"
                            in_counter += 1
                            counted_ids_in.add(track_id)
                            counted_ids_out.discard(track_id)  # Reset IN count flag

                    elif last_side == 255 and current_side == 0 and (bottom_midpoint_y - last_y > 3):  # Ensure significant movement
                        if track_id not in counted_ids_out:  # Prevent duplicate "OUT" counts
                            movement_direction = "Out"
                            out_counter += 1
                            counted_ids_out.add(track_id)
                            counted_ids_in.discard(track_id)  # Reset OUT count flag


                    last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)
                else:
                    current_pixel = mask[bottom_midpoint_y, bottom_midpoint_x]
                    current_side = 255 if current_pixel == 255 else 0
                    last_positions[track_id] = (bottom_midpoint_x, bottom_midpoint_y, current_side)

                # Save tracking data to list
                if movement_direction in ["In", "Out"]:  # Only store meaningful movements
                    tracking_data.append([track_id, movement_direction, timestamp])

        # Calculate FPS
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Display FPS, In, Out on Frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"In: {in_counter}", (w - 400, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Out: {out_counter}", (w - 400, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw threshold line before displaying/writing the frame
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)  # Red threshold line

        # Write frame to output video
        out_video.write(frame)

        # Show processed frame
        cv2.imshow("Video Processing", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out_video.release()  # Save video
    cv2.destroyAllWindows()
    print (tracking_data)
    # Save tracking data to CSV
    df = pd.DataFrame(tracking_data, columns=["Person_ID", "Direction", "Timestamp"])
    df.to_csv(csv_output_path, index=False)
    
    print(f"Processing complete. Tracking data saved to {csv_output_path}")
    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
        mask_path = config["mask_path"]
        model_path = config["model_path"]
        video_path = config["video_path"]

    output_folder = "C:/tracking/footfall_counter/Video/processed_results_vid2"
    os.makedirs(output_folder, exist_ok=True)

    process_video(video_path, mask_path, model_path, output_folder)
