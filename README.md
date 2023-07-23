# Face Detection from Webcam and Video Feed

## Author
Artificial Nouveau

## Objective
This repository contains two Python scripts for performing face detection:

1. `webcam_face_detection.py` uses a webcam as input.
2. `video_face_detection.py` uses a video file as input.

Both scripts detect faces in the input feed, save each unique face in a separate image file in a specified output directory, and print a timestamp on each face image.

This repository contains two Python scripts: `webcam_face_detection.py` and `video_face_detection.py` which are used to detect and process faces in real-time from a webcam feed and a pre-recorded video respectively.

## Prerequisites
These scripts require the following libraries to be installed:

- `opencv-python` for video processing.
- `face_recognition` for face detection.
- `Pillow` for image manipulation.
- `numpy` for array manipulations.
- `tqdm` for progress bar.
- `argparse` for command-line arguments.

You can install them using pip:
```bash
pip install opencv-python face_recognition Pillow numpy tqdm argparse
```

## Usage
You can run the webcam face detection script as follows:

```bash
python webcam_face_detection.py /path/to/output/directory
```

Similarly, you can run the video face detection script as follows:

```bash
python video_face_detection.py /path/to/video/file /path/to/output/directory
```

In both cases, the last argument is the directory where the cropped face images, the merged faces image, merged faces with timestamps image and the average face image will be saved.

For the webcam script, press 'q' to stop the webcam and process the faces.

## Output
For each detected face, the scripts save an image of the face in the output directory. The image filename is the timestamp at which the face was detected. 

The scripts also generate three special images in the output directory:

- `merged_faces.jpg` which is an image of all the detected faces aligned and merged into one image, with a maximum of 5 faces per row. 
- `merged_faces_timestamps.jpg` which is the same as `merged_faces.jpg` but with the timestamp written under each face.
- `average_face.jpg` which is an image of the average of all detected faces.

## Note
These scripts assume the faces in the input video or webcam feed are of different individuals. Faces of the same individual appearing multiple times are considered duplicates and only the first occurrence is processed.

Also, due to the nature of face detection and recognition algorithms, the scripts may not detect all faces in the video or webcam feed, especially if the faces are small, partially occluded, or turned away from the camera. 

Finally, remember to adjust the variables `FACE_HEIGHT`, `FACE_WIDTH`, and `THRESHOLD` in both scripts if needed, to suit your specific needs and conditions.
