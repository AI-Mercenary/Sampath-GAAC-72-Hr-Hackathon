from transformers import pipeline
from models.video_audio_extraction import audio_clip_dict

def generate_summary_from_audio(title, description):
    summarizer = pipeline("summarization", model="meta-llama/Llama-3.3-70B-Instruct")
    audio_texts = [text for _, text in audio_clip_dict.values()]
    input_text = f"Title: {title}\nDescription: {description}\nAudio Texts: {' '.join(audio_texts)}"
    summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']
