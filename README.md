# Face Detection and Cropping from Webcam and Video Files

This repository contains two Python scripts that use OpenCV, dlib, and the face_recognition library to detect faces from a webcam feed or video file. The detected faces are then cropped, timestamped, and saved in a folder named after the current date. At the end of the process, all faces are combined into a single image.

## Prerequisites

- Python 3.7 or later.
- The following Python libraries: OpenCV, dlib, face_recognition, PIL (Pillow), numpy. These can be installed via pip:

```sh
pip install opencv-python dlib face_recognition Pillow numpy
```

- A TrueType Font (TTF) file for writing timestamps onto images. The scripts assume the DejaVuSans-Bold.ttf font file is located at /usr/share/fonts/truetype/dejavu/. If you have a different TTF file or it's located at a different path, you should adjust the font variable in the scripts accordingly.

```python
font = ImageFont.truetype("<path_to_your_font_file>", 14)
```

## Usage

### 1. Face Detection and Cropping from Webcam Feed

Run the script `webcam_face_detection.py`. The script will start the webcam feed and begin to detect faces. Each detected face will be saved in a new directory named after the current date. Press 'q' to stop the script. At the end, a single image containing all detected faces will be saved in the directory.

### 2. Face Detection and Cropping from Video File

Before running the script `video_face_detection.py`, replace the `'video.mp4'` placeholder in the script with the path to your video file.

Run the script with `python video_face_detection.py`. The script will read the video file frame by frame and detect faces, just like the webcam script. Detected faces will be saved in a new directory named after the current date. At the end, a single image named `<video_filename>_all_faces.png` containing all detected faces will be saved in the directory.

###Optional: YouTube Video Downloader

This repository contains a Python script to download YouTube videos based on their URLs.

The script uses the `pytube` library to fetch and download the YouTube video. The video is downloaded in its highest available resolution.

## Prerequisites

```sh
pip install pytube
```

## Usage

To use the script, run it from the command line and pass the URL of the YouTube video you want to download as the first command line argument, and the output path as the second command line argument:

```sh
python download_video.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' '/path/to/download/directory'  # replace with your youtube link and desired download path
```

The video will be downloaded to the directory specified by the output path argument.

## Important Note

The face recognition used in these scripts is not 100% accurate and can have a harder time recognizing the same person in different lighting conditions, from different angles, with different facial expressions, etc. These scripts are intended for educational and personal use and may not be suitable for high-accuracy face recognition requirements.
