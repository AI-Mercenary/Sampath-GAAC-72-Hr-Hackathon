import cv2
import yt_dlp
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline
import os
from models.summurisation import generate_summary_from_audio
from models.video_audio_extraction import process_audio_clips, process_video_frames #, audio_clip_dict

def download_video(link, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    print(f"Video downloaded to {output_path}")

def extract_video_metadata(link):
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
    title = info.get("title", "No title available")
    description = info.get("description", "No description available")
    return title, description

def extract_keywords(title, description):
    classifier = pipeline("feature-extraction", model="distilbert-base-uncased")
    text = f"{title} {description}"
    keywords = classifier(text)
    keywords = [word for word in keywords if word.isalpha()]
    with open("processed/keywords.txt", "w") as f:
        f.write("\n".join(keywords))
    print("Keywords extracted and saved to processed/keywords.txt")

def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 30 == 0:  # Capture every 30th frame
            frames.append(frame)

    cap.release()
    print("Video processing completed.")
    return frames

def generate_pdf(summary, key_frames, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Video Summary Report")
    c.drawString(100, 730, f"Summary: {summary}")
    y_position = 710

    for frame in key_frames:
        frame_path = "processed/frame_image.jpg"
        Image.fromarray(frame).save(frame_path)
        c.drawImage(frame_path, 100, y_position, width=200, height=150)
        y_position -= 160
        if y_position < 100:  # Add a new page if space runs out
            c.showPage()
            y_position = 750

    c.save()
    print(f"PDF saved as {output_pdf}")

def process_video(link, output_pdf):
    video_path = "./raw/downloaded_video.mp4"
    download_video(link, video_path)
    title, description = extract_video_metadata(link)
    extract_keywords(title, description)
    process_audio_clips(video_path, "Qwen2-7B-LLM-F16", title)
    summary = generate_summary_from_audio(title, description)
    frames = process_video_stream(video_path)
    if not frames:
        print("No frames captured from video.")
        return
    generate_pdf(summary, frames, output_pdf)

if __name__ == "__main__":
    video_link = "https://www.youtube.com/watch?v=lGgIhxYuSHM"
    output_pdf_path = "./processed/video_summary_report.pdf"
    os.makedirs("./processed", exist_ok=True)
    process_video(video_link, output_pdf_path)
