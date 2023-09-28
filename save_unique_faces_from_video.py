#%%
import cv2
import numpy as np
import face_recognition
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
from contextlib import redirect_stdout
import os
import glob

detector = MTCNN()

def get_face_encodings_from_frame(frame):
    rgb_frame = frame[:, :, ::-1]
    
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            detected_faces = detector.detect_faces(rgb_frame)
    
    if not detected_faces:
        return []
    
    face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in detected_faces]
    face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
    
    return face_encodings


def find_and_collate_faces(video_path, output_folder, N=10):
    video_basename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_basename)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    known_encodings = []
    frames_dict = {}

    for current_frame in tqdm(range(0, total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % N == 0:
            encodings = get_face_encodings_from_frame(frame)
            for encoding in encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    frames_dict[first_match_index].append(frame)
                else:
                    known_encodings.append(encoding)
                    frames_dict[len(known_encodings) - 1] = [frame]

    cap.release()

    for idx, frames in frames_dict.items():
        output_video_path = f"{output_folder}/{video_name}_person_{idx}.mp4"
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, (frames[0].shape[1], frames[0].shape[0]))
        print(output_video_path)
        for frame in frames:
            out.write(frame)
        out.release()

    print(f"Found and saved videos for {len(known_encodings)} unique faces.")


def process_all_videos_in_folder(input_folder, output_folder):
    # Search for all .mp4 files in the input_folder
    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))

    # Loop through each video file and process it
    for video_file in video_files:
        print(f"Processing {video_file}")
        find_and_collate_faces(video_file, output_folder)

#%%
# Run this if you want to save faces from a single fvideo
# video_path = "input_file"
# output_folder = "output_folder"
# find_and_collate_faces(video_path, output_folder)

#%%
input_folder = 'input_folder'
output_folder = 'output_folder'
process_all_videos_in_folder(input_folder, output_folder)

