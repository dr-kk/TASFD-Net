from utils.video_utils import *

video="v2.mp4"

frames=read_video(video)

print("Frames :",len(frames))

frames=resize_frames(frames)

sampled = frames

def create_clips(frames,
                 clip_length=16,
                 stride=4):

    clips=[]

    for i in range(0,
                   len(frames)-clip_length+1,
                   stride):

        clips.append(frames[i:i+clip_length])

    return clips

clips = create_clips(sampled, clip_length=16, stride=4)

print("Clips :",len(clips))