from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load the YOLOv8 model (using a large model for better detection)
model = YOLO('yolov8s.pt')  # Change to 'yolov8n.pt' or other versions if needed

# Path to the input video
video_path = "Z:/object-detection-yolo/files/traffic-mini.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's total frame count, frame width, height, and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Total Frames: {total_frames}, FPS: {fps}, Resolution: {frame_width}x{frame_height}")

# Create the output folder if it doesn't exist
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Generate a unique filename with the current timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"processed_video_{current_time}.mp4"
output_path = os.path.join(output_folder, output_filename)

# Define the codec and output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize frame counter
processed_frames = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    # Run YOLOv8 object detection on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Increment frame counter and calculate progress
    processed_frames += 1
    progress = int((processed_frames / total_frames) * 100)
    print(f"Processing: {progress:.2f}% complete", end="\r")  # Print progress in the same line

    # Display the frame (optional)
    cv2.imshow(' Object Detection', annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nOutput video saved to {output_path}")
