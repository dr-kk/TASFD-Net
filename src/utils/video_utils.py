import cv2
import os
import numpy as np

#############################################################
# Read Video
#############################################################

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    return frames


#############################################################
# Resize Frames
#############################################################

def resize_frames(frames, size=224):

    output=[]

    for frame in frames:

        frame=cv2.resize(frame,(size,size))

        output.append(frame)

    return output


#############################################################
# Temporal Sampling
#############################################################

def temporal_sampling(frames, fps=1, original_fps=30):

    interval=max(1,int(original_fps/fps))

    sampled=[]

    for i in range(0,len(frames),interval):

        sampled.append(frames[i])

    return sampled


#############################################################
# Create Clips
#############################################################

def create_clips(frames,
                 clip_length=16,
                 stride=4):
    """
    Create overlapping clips using a sliding window.
    """

    clips = []

    if len(frames) < clip_length:
        return []

    for i in range(0, len(frames) - clip_length + 1, stride):
        clips.append(frames[i:i + clip_length])

    return clips

#############################################################
# Normalize
#############################################################

def normalize_clip(clip):

    clip=np.array(clip).astype(np.float32)

    clip/=255.0

    return clip


#############################################################
# Save Frames
#############################################################

def save_frames(frames,
                folder):

    os.makedirs(folder,
                exist_ok=True)

    for i,frame in enumerate(frames):

        frame=cv2.cvtColor(frame,
                           cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(folder,
                    f"{i:05d}.jpg"),
                    frame)