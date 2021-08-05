import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import argparse


def extract(sound_list, video_path, save_path):
    for audio_id in sound_list:
        name = os.path.join(video_path, audio_id)
        audio_name = audio_id[:-4] + '.wav'
        exist_lis = os.listdir(save_path)
        if audio_name in exist_lis:
            print("already exist!")
            continue
        try:
            video = VideoFileClip(name)
            audio = video.audio
            audio.write_audiofile(os.path.join(save_path, audio_name), fps=16000)
            print("finish video id: " + audio_name)
        except:
            print("cannot load ", name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/LLP_dataset/audio')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/LLP_dataset/video')
    args = parser.parse_args()

    sound_list = os.listdir(args.video_pth)
    extract(sound_list, args.video_path, args.out_dir)
