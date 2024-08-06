# command: python blip2.py --fps 1
import os
import math
import numpy as np
import cv2
import yaml
import torch
import json
from decord import VideoReader
import argparse

from tqdm import tqdm
from projects.BLIP2.demo_blip2_caption import select_device, VisualizationBLIP2Demo
from projects.Omnivl.data.utils import load_video_from_path_decord
from projects.Omnivl.demo_omnivl_video import OmniVL_VideoDemo
from projects.Omnivl.demo_omnivl_image import OmniVLVQAPredictor

config_path="paths.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
frames_root = config["frames_root"]
video_root = config["video_root"]
framecaps_root = config["framecaps_root"]

if not os.path.exists(framecaps_root):
    os.makedirs(framecaps_root)

def get_duration(video_path):
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    return round(len(vr) / fps, 2), fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--model_name", default="OPT2.7B", help="path to configuration file."
    )
    parser.add_argument(
        "--fps", type=int, default=1, help="path to configuration file."
    )
    args = parser.parse_args()
    model_name = args.model_name
    spe_fps = args.fps
    
    demo = VisualizationBLIP2Demo(model_name)
    
    #videos_dir = "/data1/v-junkewang/activitynet/frames"
    #mp4s_dir = "/data1/v-junkewang/activitynet/videos"
    videos_dir = frames_root #"demo_frames"
    mp4s_dir = video_root #"demo_videos"
    videos = os.listdir(videos_dir)

    model_name_ = model_name.replace(" ", "_")
    #caption_dir = f"/data1/v-junkewang/activitynet/{model_name_}"
    caption_dir = framecaps_root #f"demo_captions/{model_name_}"
    os.makedirs(caption_dir, exist_ok=True)
    
    for vid in tqdm(videos):

        json_path = os.path.join(caption_dir, vid+".json")
        if os.path.exists(json_path):
            print("%s exists." % vid)
            continue

        vid_path = os.path.join(videos_dir, vid)
        mp4_path = os.path.join(mp4s_dir, vid+".mp4")
        duration, fps = get_duration(mp4_path)
                
        default_num_frms = len(os.listdir(vid_path))
        
        resample_indices = np.linspace(
            0, default_num_frms - 1, int(duration) * spe_fps, dtype=int
        )
        
        #print(default_num_frms, duration, fps, resample_indices)
        
        caption_list = demo.run_on_video(
            vid_path, resample_indices
        )
        
        # print(duration, fps, len(caption_list), caption_list[0])
        save_item = {
            "video_name": vid,
            "fps": fps,
            "duration": duration,
            "captions": caption_list
        }
        
        with open(json_path, "w") as f:
            json.dump(save_item, f)