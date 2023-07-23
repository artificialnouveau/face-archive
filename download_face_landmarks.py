import os
import urllib.request
import bz2

# Define the URL where the model can be downloaded.
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Define the local file path where you want to store the downloaded file.
file_path = "shape_predictor_68_face_landmarks.dat.bz2"

# Check if the file already exists.
if not os.path.isfile(file_path):
    # The file does not exist, so let's download it.
    print("File not found, starting download...")
    urllib.request.urlretrieve(url, file_path)
    print(f"File downloaded to {file_path}")
    with open(file_path[:-4], 'wb') as new_file, bz2.BZ2File(file_path, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)
    print(f"File decompressed to {file_path[:-4]}")
else:
    print("File already exists.")
