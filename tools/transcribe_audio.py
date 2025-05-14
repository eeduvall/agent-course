import wave
import json
import numpy as np
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from io import BytesIO
from langchain.tools import Tool
from tools.utils.file_api_handler import download_task_file

def transcribe_audio_from_task(task_id):
    """Transcribe an audio file directly from the task_id without using file_downloader tool."""
    try:
        # Get the file directly using download_task_file
        filename, file_content = download_task_file(task_id)
        
        # Check if we got a valid audio file by extension
        file_extension = filename.lower().split('.')[-1] if filename else ''
        if file_extension != 'mp3':
            return f"Error: File is not an MP3 audio file. File type: {file_extension}"
        
        return transcribe_audio_from_file(file_content)
        
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def transcribe_audio_from_binary(file_content):
    # Convert MP3 binary data to WAV format
    audio = AudioSegment.from_file(BytesIO(file_content), format="mp3")
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Save as WAV in memory
    buffer = BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    # Load audio into Wave format
    with wave.open(buffer, 'rb') as wf:
        model = Model("model/vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, wf.getframerate())

        results = []
        while (data := wf.readframes(4000)):
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                results.append(result.get("text", ""))
        
        return {"transcription": " ".join(results)}

transcribe_audio_tool = Tool(
    name="Transcribe Audio",
    func=transcribe_audio_from_task,
    description="Transcribe text from an audio file. Provide the task_id to analyze the audio. Do NOT use the file_downloader tool before this one.",
)