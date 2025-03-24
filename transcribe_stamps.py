import whisper

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

# Example usage
audio_path = "temp_audio.wav"
transcription_with_timestamps = transcribe_audio_with_whisper(audio_path)
for item in transcription_with_timestamps:
    print(f"Start: {item['start_time']} - End: {item['end_time']} - Text: {item['text']}")
