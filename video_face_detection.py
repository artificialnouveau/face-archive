import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import argparse
from tqdm import tqdm
from scipy.spatial import distance
from skimage.io import imread_collection
import os

FACE_HEIGHT = 100
FACE_WIDTH = 100
THRESHOLD = 0.6

def save_face(image, top, right, bottom, left, output_dir, timestamp):
    face_image = image[top:bottom, left:right]

    # Convert to PIL Image
    pil_image = Image.fromarray(face_image)

    # Create a new image with black background
    background = Image.new('RGB', (pil_image.width, pil_image.height + 30), 'black')

    # Paste the original image into the background image
    background.paste(pil_image, (0,0))

    # Prepare to draw on the new image
    draw = ImageDraw.Draw(background)

    # Write timestamp on the new black area at the bottom of the image
    draw.text((0, pil_image.height), timestamp, fill="white")

    # Save the new image
    # filename = f"{output_dir}/{timestamp}.jpg"
    # background.save(filename)

    return face_image

def merge_faces(face_images, output_dir):
    n = len(face_images)
    ncols = 5
    nrows = n // ncols
    if n % ncols != 0:
        nrows += 1

    merged_image = np.zeros((nrows * FACE_HEIGHT, ncols * FACE_WIDTH, 3), dtype=np.uint8)

    for i, face_image in enumerate(face_images):
        row = i // ncols
        col = i % ncols

        face_image_resized = cv2.resize(face_image, (FACE_WIDTH, FACE_HEIGHT))
        merged_image[row * FACE_HEIGHT : (row + 1) * FACE_HEIGHT, col * FACE_WIDTH : (col + 1) * FACE_WIDTH] = face_image_resized

    merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/merged_faces.jpg", merged_image_bgr)

    merged_image = cv2.cvtColor(merged_image_bgr, cv2.COLOR_BGR2RGB)
    return merged_image

def is_new_face(face_encoding, known_face_encodings):
    if len(known_face_encodings) == 0:
        return True

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return np.min(face_distances) > THRESHOLD

def main(video_path, output_dir):
    video_capture = cv2.VideoCapture(video_path)
    known_face_encodings = []
    face_images = []
    frame_number = 0

    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=length, ncols=70) as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_number += 1

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                if is_new_face(face_encoding, known_face_encodings):
                    known_face_encodings.append(face_encoding)

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    face_image = save_face(rgb_frame, top, right, bottom, left, output_dir, timestamp)
                    face_images.append(face_image)

            pbar.update(1)

    merge_faces(face_images, output_dir)
    video_capture.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save unique faces from a video")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_dir", help="Directory to save the cropped faces")
    args = parser.parse_args()

    main(args.video_path, args.output_dir)
