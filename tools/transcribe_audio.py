import wave
import json
import numpy as np
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from io import BytesIO
from langchain.tools import Tool
# Load your binary MP3 audio file
def transcribe_audio(binary_audio):

    result = get_task_file(task_id)
    
    # Check if we got a valid image
    if result.get('file_type') == 'mp3' and 'content' in result:
        audio = result['content'] 
        
        # Convert MP3 binary data to WAV format
        audio = AudioSegment.from_file(BytesIO(binary_audio), format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Save as WAV in memory
        buffer = BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # Load audio into Wave format
        with wave.open(buffer, 'rb') as wf:
            model = Model("model")  # Ensure you've downloaded a Vosk model
            recognizer = KaldiRecognizer(model, wf.getframerate())

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    results.append(result.get("text", ""))

            with open("audio.txt", "w") as file:
                file.write(" ".join(results))
            return " ".join(results)
    else:
        return f"Error: Could not process image for task {task_id}. File type: {result.get('file_type')}. Stop processing immediately and report an error."

transcribe_audio_tool = Tool(
    name="Transcribe Audio",
    func=transcribe_audio,
    description="Transcribe text from a audio file. Do NOT use the file_downloader tool before this one.",
)