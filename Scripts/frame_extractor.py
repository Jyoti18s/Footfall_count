import cv2
import os
import json

# Function to extract frames 
def FrameCapture(path): 

	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Saves the frames with frame-count 
		cv2.imwrite("frame%d.jpg" % count, image)
		break



# # Function to extract frames as thumbnails
# def extract_thumbnails(video_path, output_folder, thumbnail_size=(200, 200), frame_interval=30):
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Open the video file
#     vidObj = cv2.VideoCapture(video_path)
#     count = 0
#     frame_id = 0

#     while True:
#         success, frame = vidObj.read()
#         if not success:
#             break  # Stop if no more frames

#         # Extract a frame at specified interval
#         if frame_id % frame_interval == 0:
#             # Resize to thumbnail size
#             thumbnail = cv2.resize(frame, thumbnail_size)

#             # Save the frame as an image
#             output_path = os.path.join(output_folder, f"thumbnail_{count}.jpg")
#             cv2.imwrite(output_path, thumbnail)
#             print(f"Saved: {output_path}")

#             count += 1  # Increment thumbnail count

#         frame_id += 1  # Increment frame index

#     vidObj.release()
#     print("Frame extraction completed.")

# Driver Code
if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
        video_path = config["video_path"]
        output_folder = config["frames_folder"]
    FrameCapture(video_path)
        

    # extract_thumbnails(video_path, output_folder, thumbnail_size=(200, 200), frame_interval=30)



# # Driver Code 
# if __name__ == '__main__':  
# 	FrameCapture('C:/tracking/footfall_counter/Video/test_videos/20250319113426716_446eebe8dbfb4eff974b964a0ee21c32_B73713197.mp4') 
