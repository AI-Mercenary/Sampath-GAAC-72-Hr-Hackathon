from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def read_transcription_file(transcription_path):
    with open(transcription_path, 'r') as file:
        transcription = file.read()
    return transcription

def summarize_text(transcription):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcription, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def generate_pdf_from_summary(summary, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Summary Report")
    text_lines = summary.split('. ')
    y_position = 720
    for line in text_lines:
        if y_position < 50:  
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750
        c.drawString(100, y_position, line.strip() + '.')
        y_position -= 20
    c.save()
    print(f"PDF saved at {output_pdf_path}")

def process_transcription_to_summary(transcription_path, output_pdf_path):
    transcription = read_transcription_file(transcription_path)
    summary = summarize_text(transcription)
    generate_pdf_from_summary(summary, output_pdf_path)

if __name__ == "__main__":
    transcription_file_path = "./processed/audio_transcription.txt"
    output_pdf_path = "./outputs/summary_report.pdf"
    process_transcription_to_summary(transcription_file_path, output_pdf_path)
