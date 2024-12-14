import os
import googleapiclient.discovery
import googleapiclient.errors
import json

def get_youtube_service(api_key):
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key)
    return youtube

def get_video_stats(youtube, video_id):
    request = youtube.videos().list(
        part="statistics, snippet",
        id=video_id
    )
    response = request.execute()
    return response

def get_suggested_videos(youtube, video_id):
    request = youtube.search().list(
        part="snippet",
        relatedToVideoId=video_id,
        type="video",
        maxResults=10
    )
    response = request.execute()
    return response

def save_clips_to_file(clips, filename):
    with open(filename, 'w') as f:
        json.dump(clips, f)
    print(f"Clips saved to {filename}")

def process_youtube_video(api_key, video_link):
    video_id = video_link.split("v=")[-1]
    youtube = get_youtube_service(api_key)
    
    video_stats = get_video_stats(youtube, video_id)
    suggested_videos = get_suggested_videos(youtube, video_id)
    
    clips = {
        "video_stats": video_stats,
        "suggested_videos": suggested_videos
    }
    
    os.makedirs("./processed", exist_ok=True)
    save_clips_to_file(clips, "./processed/clips.json")

if __name__ == "__main__":
    api_key = "YOUR_YOUTUBE_API_KEY"
    video_link = "https://www.youtube.com/watch?v=lGgIhxYuSHM"
    process_youtube_video(api_key, video_link)
