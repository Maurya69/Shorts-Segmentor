import whisper
import re
import csv
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip
from gen import gen
import pandas as pd
def convert_mp4_to_wav(mp4_path, wav_path):
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(wav_path)

def transcribe_audio_with_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    segments = result['segments']
    transcription_with_timestamps = []

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        transcription_with_timestamps.append({"start_time": start_time, "end_time": end_time, "text": text})
    
    return transcription_with_timestamps

def split_sentences_with_timestamps(transcription_with_timestamps):
    sentence_pattern = re.compile(r'[^.!?]+[.!?]')
    sentences_with_timestamps = []

    for item in transcription_with_timestamps:
        start_time = item['start_time']
        text = item['text']
        sentences = sentence_pattern.findall(text)
        sentence_start_time = start_time

        for sentence in sentences:
            sentence_end_time = sentence_start_time + (len(sentence) / len(text)) * (item['end_time'] - item['start_time'])
            sentences_with_timestamps.append({"start_time": sentence_start_time, "end_time": sentence_end_time, "text": sentence.strip()})
            sentence_start_time = sentence_end_time

    return sentences_with_timestamps

def write_to_csv(sentences_with_timestamps, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["start_time", "end_time", "text"])
        writer.writeheader()
        for sentence in sentences_with_timestamps:
            writer.writerow(sentence)

