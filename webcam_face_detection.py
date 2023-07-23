import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import argparse
from tqdm import tqdm
from scipy.spatial import distance

FACE_HEIGHT = 100
FACE_WIDTH = 100
THRESHOLD = 0.6

def save_face(image, top, right, bottom, left, output_dir, timestamp):
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)

    background = Image.new('RGB', (pil_image.width, pil_image.height + 30), 'black')
    background.paste(pil_image, (0,0))

    draw = ImageDraw.Draw(background)
    draw.text((0, pil_image.height), timestamp, fill="white")

    # filename = f"{output_dir}/{timestamp}.jpg"
    # background.save(filename)

    return face_image, timestamp

def merge_faces(face_images, face_timestamps, output_dir):
    n = len(face_images)
    ncols = 5
    nrows = n // ncols
    if n % ncols != 0:
        nrows += 1

    merged_image = np.zeros((nrows * (FACE_HEIGHT+30), ncols * FACE_WIDTH, 3), dtype=np.uint8)
    merged_image_timestamps = np.zeros((nrows * (FACE_HEIGHT+30), ncols * FACE_WIDTH, 3), dtype=np.uint8)

    for i, (face_image, timestamp) in enumerate(zip(face_images, face_timestamps)):
        row = i // ncols
        col = i % ncols

        face_image_resized = cv2.resize(face_image, (FACE_WIDTH, FACE_HEIGHT))
        merged_image[row * (FACE_HEIGHT+30) : (row * (FACE_HEIGHT+30)) + FACE_HEIGHT, col * FACE_WIDTH : (col + 1) * FACE_WIDTH] = face_image_resized
        merged_image_timestamps[row * (FACE_HEIGHT+30) : (row * (FACE_HEIGHT+30)) + FACE_HEIGHT, col * FACE_WIDTH : (col + 1) * FACE_WIDTH] = face_image_resized

        cv2.putText(merged_image_timestamps, timestamp, (col * FACE_WIDTH, (row * (FACE_HEIGHT+30)) + FACE_HEIGHT + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite(f"{output_dir}/merged_faces.jpg", cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/merged_faces_timestamps.jpg", cv2.cvtColor(merged_image_timestamps, cv2.COLOR_RGB2BGR))

def average_face(face_images, output_dir):
    avg_image = np.zeros((FACE_HEIGHT, FACE_WIDTH, 3))

    for face_image in face_images:
        resized_image = cv2.resize(face_image, (FACE_WIDTH, FACE_HEIGHT))
        avg_image += resized_image

    avg_image = avg_image / len(face_images)
    cv2.imwrite(f"{output_dir}/average_face.jpg", cv2.cvtColor(avg_image.astype('uint8'), cv2.COLOR_RGB2BGR))

def is_new_face(face_encoding, known_face_encodings):
    if len(known_face_encodings) == 0:
        return True

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    return np.min(face_distances) > THRESHOLD

def main(output_dir):
    video_capture = cv2.VideoCapture(0)
    known_face_encodings = []
    face_images = []
    face_timestamps = []
    frame_number = 0

    print("Press 'q' to stop webcam and process faces.")

    while True:
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
                face_image, timestamp = save_face(rgb_frame, top, right, bottom, left, output_dir, timestamp)
                face_images.append(face_image)
                face_timestamps.append(timestamp)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    merge_faces(face_images, face_timestamps, output_dir)
    average_face(face_images, output_dir)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save unique faces from a video")
    parser.add_argument("output_dir", help="Directory to save the cropped faces")
    args = parser.parse_args()

    main(args.output_dir)
