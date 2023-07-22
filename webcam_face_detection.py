import sys
import cv2
import face_recognition
import os
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont

# Declare output directory as a global variable
output_dir = ""
known_face_encodings = []

def save_face(face_image):
    global output_dir
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    img_path = os.path.join(output_dir, timestamp_str + '.jpg')

    # Save the image
    face_image_pil = Image.fromarray(face_image)
    draw = ImageDraw.Draw(face_image_pil)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
    draw.rectangle(((0, face_image.shape[0] - 30), (face_image.shape[1], face_image.shape[0])), fill="black")
    draw.text((10, face_image.shape[0] - 30), timestamp_str, font=font, fill=(255, 255, 255, 128))

    face_image_pil.save(img_path)


def process_frame(frame):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True not in matches:
            known_face_encodings.append(face_encoding)
            save_face(frame[top:bottom, left:right])

def main(output_directory):
    global output_dir
    output_dir = output_directory

    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        process_frame(frame)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1])
