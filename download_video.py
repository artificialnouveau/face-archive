import sys
from pytube import YouTube

def download_video(youtube_url, output_path):
    youtube = YouTube(youtube_url)
    video = youtube.streams.get_highest_resolution()
    video.download(output_path)

# Use the function with command-line arguments
download_video(sys.argv[1], sys.argv[2])  # the first and second command line arguments
