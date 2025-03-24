# import whisper
# def transcribe(audio_file):
#     model=whisper.load_model("base")
#     result=model.transcribe(audio_file)
#     with open("trans.txt","w") as f:
#         f.write(result["text"])

# transcribe("temp_audio.wav")

import speech_recognition as sr
from pydub import AudioSegment
import os

# Load the video file
video = AudioSegment.from_file("video.mp4", format="mp4")
audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
audio.export("audio.wav", format="wav")