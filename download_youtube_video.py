#%%
import sys
import os
import re
from pytube import YouTube
import subprocess


#%%
def sanitize_filename(filename):
    # Remove special characters and spaces from the filename
    return re.sub(r'[^a-zA-Z0-9]', '', filename)

def sanitize_filename(filename):
    # Remove special characters and spaces from the filename
    return re.sub(r'[^a-zA-Z0-9]', '', filename)

def download_youtube_video_dl(link):
    # Using youtube-dl to get the video title
    title = subprocess.getoutput(f'youtube-dl --get-title {link}')
    sanitized_title = sanitize_filename(title)

    # Set file names
    video_filename = f"{sanitized_title}.mp4"
    audio_wav_filename = f"{sanitized_title}_audio.wav"

    # Downloading the highest resolution video in mp4 format using youtube-dl
    subprocess.run(['youtube-dl', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]', link, '-o', video_filename])
    
    # Extracting and converting audio to .wav format using youtube-dl and ffmpeg
    subprocess.run(['youtube-dl', '-x', '--audio-format', 'wav', link, '-o', audio_wav_filename])

    # Return file paths
    video_path = os.path.abspath(video_filename)
    audio_wav_path = os.path.abspath(audio_wav_filename)

    return video_path, audio_wav_path


def download_youtube_video_pytube(link):
    yt = YouTube(link)

    # Get sanitized title
    sanitized_title = sanitize_filename(yt.title)

    # Set file names
    video_filename = f"{sanitized_title}.mp4"
    audio_filename = f"{sanitized_title}_audio.mp4"
    audio_wav_filename = f"{sanitized_title}_audio.wav"

    # Downloading the highest resolution video in mp4 format
    video_stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
    video_stream.download(filename=video_filename)

    # Downloading audio and saving it as mp4
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=audio_filename)

    # Convert audio to wav format (this requires ffmpeg)
    os.system(f"ffmpeg -i {audio_filename} {audio_wav_filename}")

    # Remove the temporary audio mp4 file
    os.remove(audio_filename)

    # Return file paths and filenames
    video_path = os.path.abspath(video_filename)
    audio_wav_path = os.path.abspath(audio_wav_filename)
    print(video_path)
    print(audio_wav_filename)

    return video_path, audio_wav_path


#%%
# run this code to run the code within python
# video_path, audio_path = download_youtube_video("https://www.youtube.com/watch?v=ID")


#%%
# run this code if you want to run the file in the terminal
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_youtube.py <youtube_link>")
        sys.exit(1)
    
    youtube_link = sys.argv[1]
    video_path, audio_path = download_youtube_video(youtube_link)
    print(f"Video saved at: {video_path}")
    print(f"Audio saved at: {audio_path}")
