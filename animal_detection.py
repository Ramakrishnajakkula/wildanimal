import cv2
import torch
import os
import kagglehub
import yt_dlp

# Load YOLO model (using YOLOv5 from PyTorch Hub)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s' for a lightweight model

# Function to download dataset using kagglehub
def download_dataset_kagglehub(dataset):
    print("Downloading dataset using kagglehub...")
    path = kagglehub.dataset_download(dataset)
    print("Dataset downloaded to:", path)
    return path

# Function to download YouTube video using yt_dlp
def download_youtube_video(url, output_path):
    print("Downloading YouTube video using yt_dlp...")
    output_file = os.path.join(output_path, "youtube_video.mp4")
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_file,
        'quiet': False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("YouTube video downloaded to:", output_file)
        return output_file
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None

# Kaggle dataset identifier
kaggle_dataset = "iamsouravbanerjee/animal-image-dataset-90-different-animals"  # Replace with desired dataset
dataset_dir = download_dataset_kagglehub(kaggle_dataset)

# Define the URL of the YouTube video
youtube_url = "https://www.youtube.com/watch?v=aDLir84eLWw&list=PLD0_KLEe08c15l7A2dawW10bBEj35e_3s&index=1"
video_dir = "videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Download the YouTube video
fallback_video = download_youtube_video(youtube_url, video_dir)

# Define the URL of the CCTV footage or fallback to the downloaded YouTube video
cctv_url = "http://your-cctv-url/stream"  # Replace with the actual CCTV stream URL

# Open the video stream
cap = cv2.VideoCapture(cctv_url)

if not cap.isOpened():
    if fallback_video:
        print("Error: Unable to open CCTV stream. Falling back to YouTube video.")
        cap = cv2.VideoCapture(fallback_video)
    else:
        print("Error: Unable to open CCTV stream and no fallback video available. Exiting.")
        exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from video source.")
        break

    # Perform animal detection
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Display the frame
    cv2.imshow("Animal Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
