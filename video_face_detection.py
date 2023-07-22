import cv2
import face_recognition
import datetime
import os
import numpy as np
from PIL import ImageDraw
from tqdm import tqdm

def process_frame(frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    return face_locations

def save_face(image, top, right, bottom, left, output_dir, timestamp):
    # Crop out the face from the frame
    face_image = image[top:bottom, left:right]

    pil_image = Image.fromarray(face_image)
    draw = ImageDraw.Draw(pil_image)
    # Use default font
    draw.text((0, face_image.shape[0] - 30), timestamp, fill="white")
    filename = f"{output_dir}/{timestamp}.jpg"
    pil_image.save(filename)

def main(video_path, output_dir):
    video_capture = cv2.VideoCapture(video_path)

    # Get total frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), "Processing"):
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Get frame timestamp
        frame_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = datetime.datetime.fromtimestamp(frame_time/1000.0).strftime("%Y_%m_%d-%H_%M_%S-%f")

        face_locations = process_frame(frame)
        for top, right, bottom, left in face_locations:
            save_face(frame, top, right, bottom, left, output_dir, timestamp)

    video_capture.release()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process and save faces from a video.')
    parser.add_argument('--video_path', type=str, help='Path to the video file.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output images.')
    args = parser.parse_args()

    main(args.video_path, args.output_dir)
