from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import whisper
import re
import csv
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip
from gen import gen
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from moviepy.video.VideoClip import TextClip
from assist import convert_mp4_to_wav
from assist import transcribe_audio_with_whisper
from assist import split_sentences_with_timestamps
from gen import gen
from gen import clip_video
from gen import clip_video_with_subtitles
import pandas as pd

audio_path = "temp.wav"
convert_mp4_to_wav("video1.mp4","temp.wav")
transcription_with_timestamps = transcribe_audio_with_whisper(audio_path)
sentences_with_timestamps = split_sentences_with_timestamps(transcription_with_timestamps)
sentences_with_timestamps=pd.DataFrame(sentences_with_timestamps)
thematic_segments=gen(sentences_with_timestamps)
input_path="video1.mp4"
output_dir="store"
subtitles_csv=pd.read_csv("transcriptions.csv")
outputs=clip_video_with_subtitles(input_path,thematic_segments,subtitles_csv=subtitles_csv,output_dir="C:\\Users\\maury\\OneDrive\\Desktop\\shorts\\store")
print(outputs)