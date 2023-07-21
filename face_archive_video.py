import cv2
import dlib
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime, timedelta

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Video file path
video_path = 'video.mp4'  # replace with your video file path

# Open video file
cap = cv2.VideoCapture(video_path)

# Today's date
today = datetime.now().strftime("%Y-%m-%d")
os.makedirs(today, exist_ok=True)

# List of all face images
all_faces = []
face_counter = 0

# Font for timestamp
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)

# List of known face encodings
known_face_encodings = []

# Video's frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

try:
    while True:
        # Capture frames from the video
        ret, img = cap.read()

        if not ret:
            break  # exit loop if no frame is read (end of video)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        for rect in faces:
            # Crop face
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            face_img = img[y:y+h, x:x+w]  # keep it in color for face_recognition

            # Compute face encoding
            face_encodings = face_recognition.face_encodings(face_img)

            if len(face_encodings) > 0:  # if a face is detected
                face_encoding = face_encodings[0]

                # Check if this face is in the list of known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True not in matches:  # if this is a new face
                    # Add the new face to the list of known faces
                    known_face_encodings.append(face_encoding)

                    # Convert to grayscale PIL Image to add timestamp
                    face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)).convert('RGB')

                    # Draw timestamp
                    timestamp = str(timedelta(seconds=(face_counter // fps)))
                    draw = ImageDraw.Draw(face_img_pil)
                    draw.rectangle([(0, face_img_pil.height - 20), (face_img_pil.width, face_img_pil.height)], fill="black")
                    draw.text((10, face_img_pil.height - 20), timestamp, fill="white", font=font)

                    # Save cropped face
                    face_img_pil.save(f"{today}/face_{face_counter}.png")

                    # Add to list of all faces
                    all_faces.append(face_img_pil)
                    face_counter += 1

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

# Save all faces in a single image
face_width = all_faces[0].width
face_height = all_faces[0].height

# Create a blank image with correct width and height
final_img_width = face_width * 5
final_img_height = face_height * ((len(all_faces) // 5) + (len(all_faces) % 5 > 0))
final_img = Image.new('RGB', (final_img_width, final_img_height))

# Iterate through face images
for i, face_img in enumerate(all_faces):
    final_img.paste(face_img, ((i % 5) * face_width, (i // 5) * face_height))

# Save final image with filename based on video filename
final_image_filename = os.path.splitext(os.path.basename(video_path))[0] + '_all_faces.png'
final_img.save(os.path.join(today, final_image_filename))
