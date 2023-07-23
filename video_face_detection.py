import cv2
import face_recognition
import datetime
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import dlib
import argparse
import matplotlib.pyplot as plt

# Prepare face detector and shape predictor (for face alignment)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define size to which faces will be resized
FACE_WIDTH = 100
FACE_HEIGHT = 100

def process_frame(frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    return face_locations

def save_face(image, top, right, bottom, left, output_dir, timestamp):
    # Crop out the face from the frame
    face_image = image[top:bottom, left:right]

    # Save face image
    pil_image = Image.fromarray(face_image)
    draw = ImageDraw.Draw(pil_image)
    # Use default font
    draw.text((0, face_image.shape[0] - 30), timestamp, fill="white")
    filename = f"{output_dir}/{timestamp}.jpg"
    pil_image.save(filename)

    return face_image

def merge_faces(face_images, output_dir):
    # Calculate number of rows and columns
    n = len(face_images)
    ncols = 5
    nrows = n // ncols
    if n % ncols != 0:
        nrows += 1

    # Create an empty image
    merged_image = np.zeros((nrows * FACE_HEIGHT, ncols * FACE_WIDTH, 3), dtype=np.uint8)

    # Paste images
    for i, face_image in enumerate(face_images):
        row = i // ncols
        col = i % ncols

        face_image_resized = cv2.resize(face_image, (FACE_WIDTH, FACE_HEIGHT))
        merged_image[row * FACE_HEIGHT : (row + 1) * FACE_HEIGHT, col * FACE_WIDTH : (col + 1) * FACE_WIDTH] = face_image_resized

    cv2.imwrite(f"{output_dir}/merged_faces.jpg", merged_image)

def average_faces(face_images, output_dir):
    # Resize all faces
    faces_resized = [cv2.resize(img, (FACE_WIDTH, FACE_HEIGHT)) for img in face_images]
    
    # Calculate average
    avg_face = np.mean(faces_resized, axis=0).astype(np.uint8)

    cv2.imwrite(f"{output_dir}/average_face.jpg", avg_face)

def main(video_path, output_dir):
    video_capture = cv2.VideoCapture(video_path)

    # Get total frames
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    face_images = []

    for _ in tqdm(range(total_frames), "Processing"):
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S-%f")
        face_locations = process_frame(frame)
        for top, right, bottom, left in face_locations:
            face_image = save_face(frame, top, right, bottom, left, output_dir, timestamp)
            face_images.append(face_image)

    video_capture.release()

    # Merge faces and create average face
    merge_faces(face_images, output_dir)
    average_faces(face_images, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and save faces from a video.')
    parser.add_argument('--video_path', type=str, help='Path to the video file.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output images.')
    args = parser.parse_args()

    main(args.video_path, args.output_dir)
