from langchain.tools import Tool
import os
import ssl
import tempfile
import uuid
from yt_dlp import YoutubeDL
from tools.transcribe_audio import transcribe_audio_from_binary

def youtube_processor(url: str, task_id: str = None):
    """
    Process a YouTube video URL to extract audio as text using yt-dlp.
    
    Args:
        url: The YouTube video URL
        task_id: Optional parameter (ignored, included for compatibility with LLM calls)
        
    Returns:
        Dictionary with transcription
    """
    try:
        print(f"Processing YouTube video to text: {url}")
        
        # Create a temporary directory to store the audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate a unique filename
            temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.mp3")
            print(f"Temp filename: {temp_filename}")
            
            # Use yt-dlp to download the audio
            print(f"Downloading audio from {url}...")
            
            # Disable SSL verification globally (not recommended for production, but useful for debugging)
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Configure yt-dlp options with SSL verification disabled
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': temp_filename,
                'quiet': False,
                'no_warnings': False,
                'nocheckcertificate': True  # Disable SSL certificate verification in yt-dlp
            }

            # Download the audio
            with YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading audio with YoutubeDL...")
                ydl.download([url])
                print(f"Download completed successfully")
            
            # Find the downloaded file
            actual_filename = temp_filename
            if not os.path.exists(actual_filename):
                # Try with a different extension (yt-dlp might add .mp3 extension)
                if os.path.exists(f"{temp_filename}.mp3"):
                    actual_filename = f"{temp_filename}.mp3"
                else:
                    # List files in directory to see what was created
                    files = os.listdir(temp_dir)
                    print(f"Files in directory: {files}")
                    # Try to find the downloaded file
                    for file in files:
                        if file.endswith(".mp3"):
                            actual_filename = os.path.join(temp_dir, file)
                            break
            
            print(f"Final audio file: {actual_filename}")
            
            # Read the file content as binary
            with open(actual_filename, 'rb') as f:
                file_content = f.read()
            
            print(f"Transcribing audio from {url}...")
            # Pass the binary content to transcribe_audio_from_binary
            transcription = transcribe_audio_from_binary(file_content)
            print("Transcription completed successfully:   ", transcription["transcription"])
            return transcription
    except Exception as e:
        print(f"Error processing YouTube video to text {url}: {str(e)}")
        return {"error": f"Error processing YouTube video to text {url}: {str(e)}"}
                
youtube_tool = Tool(
    name="youtube_processor",
    func=youtube_processor,
    description="Extract and transcribe the audio from a YouTube video. Provide the URL to analyze the YouTube link (e.g. https://www.youtube.com/watch?v=...). Use this tool when the quesiton calls for processing YouTube videos and urls."
)