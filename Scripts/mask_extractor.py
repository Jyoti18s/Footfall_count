import cv2
import numpy as np
import os
import json
# def generate_dynamic_mask(video_path, output_mask_path):
#     cap = cv2.VideoCapture(video_path)
#     fgbg = cv2.createBackgroundSubtractorMOG2()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         fgmask = fgbg.apply(frame)

#     cap.release()
#     cv2.imwrite(output_mask_path, fgmask)
#     print(f"Dynamic mask saved to {output_mask_path}")

# def generate_dynamic_mask(video_path, output_mask_path):
#     cap = cv2.VideoCapture(video_path)
#     fgbg = cv2.createBackgroundSubtractorMOG2()
    
#     fgmask = None  # Initialize fgmask
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # Stop if no more frames are available

#         fgmask = fgbg.apply(frame)  # Update fgmask in each frame

#     cap.release()

#     if fgmask is not None:
#         cv2.imwrite(output_mask_path, fgmask)
#         print(f"Dynamic mask saved to {output_mask_path}")
#     else:
#         print("Error: No frames processed, check the video file.")

def generate_dynamic_mask(image_path, output_mask_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Adaptive Thresholding to create a binary mask
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Save the mask
    cv2.imwrite(output_mask_path, mask)
    print(f"Dynamic mask saved to {output_mask_path}")

# Example Usage
if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
        mask_path = config["mask_path"]
        model_path = config["model_path"]
        video_path = config["video_path"]
        image_path = config["image_path"]
    output_folder = "C:/tracking/footfall_counter/processed_results"
    os.makedirs(output_folder, exist_ok=True)

    generate_dynamic_mask(image_path, mask_path)


# generate_dynamic_mask("input_image.jpg", "output_mask.jpg")

# image_path = "C:\\tracking\\footfall_counter\\Video\\test_videos\\thumbnail_frames\\thumbnail_0.jpg"
# video_path = "C:/tracking/footfall_counter/Video/test_videos/20250319112746526_41d555e762ab4d4fa665a2b471c982fc_B73713197.mp4"
# output_mask_path = "C:/tracking/footfall_counter/Video/test_videos/masks/mask1.png"
# generate_dynamic_mask(image_path, output_mask_path)
