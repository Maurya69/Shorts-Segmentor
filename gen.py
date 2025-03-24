from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import subprocess
def convert_video(input_path, output_path):
    # Define the FFmpeg command as a list
    command = [
        'ffmpeg', '-i', input_path, 
        '-c:v', 'libx264', 
        '-crf', '18', 
        '-c:a', 'aac', 
        '-b:a', '192k', 
        output_path
    ]
    
    # Run the command
    subprocess.run(command, check=True)
    print(f"Video conversion completed: {output_path}")
def gen(df):
    df['text'] = df['text'].str.lower().str.replace('[^\w\s]', '', regex=True)
    # Vectorize the text
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])

    # Apply LDA to identify themes
    lda = LatentDirichletAllocation(n_components=3, random_state=0)  # Adjust n_components based on expected themes
    lda.fit(X)

    # Assign topics to each sentence
    topic_distributions = lda.transform(X)
    df['topic'] = topic_distributions.argmax(axis=1)
    # Initialize variables
    chunks = []
    current_chunk = []
    current_topic = df.loc[0, 'topic']

    for index, row in df.iterrows():
        if row['topic'] == current_topic:
            current_chunk.append((row['start_time'], row['text']))
        else:
            chunks.append(current_chunk)
            current_chunk = [(row['start_time'], row['text'])]
            current_topic = row['topic']

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    sorted_chunks=sorted(chunks,key=len,reverse=True)
    top_chunks=sorted_chunks[:3]
    # Convert chunks into a suitable format
    thematic_segments = []
    for chunk in top_chunks:
        start_time = chunk[0][0]
        end_time = chunk[-1][0]
        text = " ".join([sentence[1] for sentence in chunk])
        thematic_segments.append({'start_time': start_time, 'end_time': end_time, 'text': text})

    # Save or use thematic_segments for further processing
    return thematic_segments


def clip_video(input_video_path, thematic_segments, output_dir):
    """
    Clips video based on thematic segments, making each segment one minute long.

    Parameters:
    - input_video_path: Path to the input video file.
    - thematic_segments: List of dictionaries containing 'start_time' and 'text'.
    - output_dir: Directory where the output clips will be saved.

    Output:
    - List of paths to the saved video clips.
    """
    # Load the video file
    video = VideoFileClip(input_video_path)
    
    output_paths = []
    
    for i, segment in enumerate(thematic_segments):
        # Extract start time in seconds
        start_time = segment['start_time']
        
        # Calculate end time (one minute later)
        end_time = start_time + 60  # 60 seconds = 1 minute
        
        # Make sure the end_time does not exceed the video duration
        end_time = min(end_time, video.duration)
        
        # Create a subclip
        subclip = video.subclip(start_time, end_time)
        
        # Define the output path
        output_path = f"{output_dir}/clip_{i+1}.mp4"
        
        # Write the subclip to file
        subclip.write_videofile(output_path, codec='libx264')
        
        # Store the output path
        output_paths.append(output_path)
    
    # Close the original video file
    video.close()
    
    return output_paths
def generate_subtitles(subtitles_csv, start_time, end_time):
    """
    Extracts subtitles for a specific time range.

    Parameters:
    - subtitles_csv: Path to the transcription CSV file containing subtitles.
    - start_time: Start time of the video segment.
    - end_time: End time of the video segment.

    Returns:
    - List of tuples in the format [(start_time, end_time, text), ...].
    """
    print(subtitles_csv)
    # Read the transcription CSV
    df = subtitles_csv

    # Filter subtitles based on the time range
    filtered_subtitles = df[(df['start_time'] >= start_time) & (df['end_time'] <= end_time)]

    # Create a list of tuples
    subtitle_list = [
        (row['start_time'] - start_time, row['end_time'] - start_time, row['text'])
        for _, row in filtered_subtitles.iterrows()
    ]
    print(subtitle_list)
    return subtitle_list



def add_subtitles_opencv(video_path, subtitles, output_path):
    """
    Adds subtitles to a video using OpenCV.

    Parameters:
    - video_path: Path to the video file.
    - subtitles: List of tuples [(start_time, end_time, text), ...].
    - output_path: Path to save the output video with subtitles.

    Returns:
    - Path to the output video.
    """
    # Validate the video path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The file {video_path} does not exist.")

    # Open video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV failed to open the video file: {video_path}")

    # Retrieve video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Font settings for subtitles
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate timestamp for the current frame
        timestamp = frame_idx / fps

        # Check and add subtitles for the current frame
        for start, end, text in subtitles:
            if start <= timestamp <= end:
                cv2.putText(frame, text, (50, height - 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    return output_path

def clip_video_with_subtitles(input_path, thematic_segments, subtitles_csv, output_dir):
    """
    Clips thematic segments from the input video and adds subtitles to each segment.

    Parameters:
    - input_path: Path to the input video file.
    - thematic_segments: List of tuples [(start_time, end_time), ...].
    - subtitles_csv: List of tuples [(start_time, end_time, text), ...] (subtitles).
    - output_dir: Directory to save the output video clips.

    Returns:
    - List of paths to the output video clips with subtitles.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    # Load the video using VideoFileClip
    video_clip = VideoFileClip(input_path)

    for i, segment in enumerate(thematic_segments):
        print(segment)
       
        start = segment['start_time']
        end = segment['end_time']
        # Extract subclip
        subclip = video_clip.subclipped(start, end)
        subtitles=generate_subtitles(subtitles_csv, start, end)
        # Temporary path for the subclip
        temp_file = os.path.join(output_dir, f"subclip_{i}.mp4")
        subclip.write_videofile(
            temp_file, 
            codec="libx264", 
            audio_codec="aac",         # Ensure audio codec is set to AAC for compatibility
            audio_bitrate="192k",      # Set audio bitrate for better quality
            threads=4,                 # Use 4 threads for faster encoding
            ffmpeg_params=["-crf", "18", "-preset", "fast"]  # Optional: Use CRF 18 for high quality
        )


        # Output path for the final video with subtitles
        output_path = os.path.join(output_dir, f"subclip_with_subtitles_{i}.mp4")

        # Add subtitles using OpenCV
        video_with_subtitles = add_subtitles_opencv(temp_file, subtitles, output_path)

        outputs.append(video_with_subtitles)
        convert_video(input_path=output_path,output_path=os.path.join(output_dir, f"subclip_with_subtitles_fixed_{i}.mp4"))

    return outputs

