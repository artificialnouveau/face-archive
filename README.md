# Face Detection from Webcam and Video Feed

## Author
Artificial Nouveau


## Objective
This repository contains two Python scripts for performing face detection:

1. `webcam_face_detection.py` uses a webcam as input.
2. `video_face_detection.py` uses a video file as input.

Both scripts detect faces in the input feed, save each unique face in a separate image file in a specified output directory, and print a timestamp on each face image.

## Prerequisites

The scripts require Python 3 and the following packages:

- OpenCV
- face_recognition
- numpy
- Pillow

Install them using pip:

```bash
pip install opencv-python
pip install face_recognition
pip install numpy
pip install pillow
```

You should also have a TTF (TrueType Font) available for writing the timestamp. The scripts assume the `DejaVuSans-Bold.ttf` font file is located at `/usr/share/fonts/truetype/dejavu/`. If this is not the case, adjust the path accordingly in the scripts.

## Usage

### Webcam Face Detection

To run the webcam face detection script:

```bash
python webcam_face_detection.py /path/to/output/directory
```

Replace `/path/to/output/directory` with the actual path to the directory where you want to save the face images.

### Video Face Detection

To run the video face detection script:

```bash
python video_face_detection.py /path/to/video.mp4 /path/to/output/directory
```

Replace `/path/to/video.mp4` with the actual path to the video file you want to process, and `/path/to/output/directory` with the actual path to the directory where you want to save the face images.

---

Please remember to set up your environment correctly before running the scripts.
