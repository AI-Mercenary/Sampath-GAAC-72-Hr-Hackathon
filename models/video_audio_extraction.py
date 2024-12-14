# This code converts the video into to a dictionary with all the text data that is associated with and describing the video and audio of the file.

import cv2
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import moviepy.editor as mp

# Global dictionaries to store extracted data
video_frame_dict = {}
audio_clip_dict = {}

def process_video_frames(video_path, model_name, title):
    cap = cv2.VideoCapture(video_path)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp in seconds

        if frame_count % 30 == 0:  # Process every 30th frame
            inputs = tokenizer(frame, return_tensors="pt", context=title) # Is this going to raise error? the context argument
            outputs = model(**inputs)
            text_output = outputs.last_hidden_state
            video_frame_dict[timestamp] = [frame, text_output]

    cap.release()
    print("Video frames processed and stored in dictionary.")

def process_audio_clips(video_path, model_name, title):
    audio = (mp.VideoFileClip(video_path)).audio
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    duration = audio.duration
    clip_start = 0

    while clip_start < duration:
        clip_end = min(clip_start + 10, duration)  # Process 10-second clips
        audio_clip = audio.subclip(clip_start, clip_end)
        inputs = tokenizer(audio_clip.to_soundarray(), return_tensors="pt", context=title) # Is this going to raise an error? The context argument
        outputs = model(**inputs)
        text_output = outputs.last_hidden_state
        audio_clip_dict[(clip_start, clip_end)] = text_output
        clip_start = clip_end

    print("Audio clips processed and stored in dictionary.")

if __name__ == "main":
    video_path = 'raw/downloaded_video.mp4'
    process_audio_clips(video_path, aud_model_name, title)
    process_video_frames(video_path, vid_model_name, title)