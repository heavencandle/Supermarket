import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)


st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
Get started by uploading a video file in the sidebar.
"""
)

def extract_audio_from_video(video_path):
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        video_path.replace("mp4", "mp3")
    ]
    subprocess.run(command)

def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3")
    final_transcript = ""

    for file in files:
        with open(file, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            final_transcript += transcript["text"]
    
    with open(destination, "w") as file:
        file.write(final_transcript)

with st.container():
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

with st.container():
    if video:
        video_content = video.read()
        video_path = f"./cache/files/{video.name}"
        with open(video_path, "wb") as f:
            f.write(video_content)
        
        extract_audio_from_video(video_path)


        chunks_folder = "./.cache/chunks"
        with st.status("Loading video..."):
            video_content = video.read()
            video_path = f"./.cache/{video.name}"
            audio_path = video_path.replace("mp4", "mp3")
            transcript_path = video_path.replace("mp4", "txt")
            with open(video_path, "wb") as f:
                f.write(video_content)
