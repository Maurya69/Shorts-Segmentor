import whisper
whisper.audio.ffmpeg_path = r"C:\\ffmpeg\bin\\ffmpeg.exe"
whisper.audio.ffprobe_path = r"C:\\ffmpeg\bin\\ffprobe.exe"
audio=whisper.load_audio('temp_audio.wav')