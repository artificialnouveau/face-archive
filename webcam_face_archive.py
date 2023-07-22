import cv2
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# For capturing video from webcam.
cap = cv2.VideoCapture(0)

# Today's date
today = datetime.now().strftime("%Y-%m-%d")
os.makedirs(today, exist_ok=True)

# List of all face images
all_faces = []
face_counter = 0

# Font for timestamp
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)

try:
    while True:
        # Capture frames from the camera
        ret, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector(gray)

        for rect in faces:
            # Crop face
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            face_img = gray[y:y+h, x:x+w]

            # Convert to PIL Image to add timestamp
            face_img_pil = Image.fromarray(face_img).convert('RGB')
            
            # Draw timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
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

# Save final image
final_img.save(f"{today}/all_faces.png")
