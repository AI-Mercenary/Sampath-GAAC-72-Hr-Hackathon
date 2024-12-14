import yt_dlp

def download_audio_from_youtube(link, output_path):
    ydl_opts = {
        'format': 'bestaudio/best', 
        'outtmpl': output_path,  
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  
            'preferredcodec': 'wav',  
            'preferredquality': '192',  
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
    print(f"Audio downloaded and saved at {output_path}")


youtube_link = "https://www.youtube.com/watch?v=yRwQ7A6jVLk"


audio_file_path = "./audio_extracted.wav"


download_audio_from_youtube(youtube_link, audio_file_path)