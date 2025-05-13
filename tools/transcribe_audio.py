# # import pytranscript as pt
# from langchain.tools import Tool

# def transcribe_audio(audio_file):
#     # convert mp3 file to wav                                                        
#     sound = AudioSegment.from_mp3(audio_file)
#     wav_file = sound.export("transcript.wav", format="wav")
#     return pt.transcribe(wav_file, model="vosk-model-en-us-aspire-0.2", max_size=None)


# transcribe_tool = Tool(
#     name="Transcribe Audio",
#     func=transcribe_audio,
#     description="Transcribe text from a audio file.",
# )