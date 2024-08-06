# command: python whisper_generate.py --gpus 8
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os, time

import yaml
import whisper
import torch

config_path="paths.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
video_root = config["video_root"]
ASR_root = config["ASR_root"]

if not os.path.exists(ASR_root):
    os.makedirs(ASR_root)

def train_on_gpu(gpu_id, vid):

    #判断文件是否存在
    ASR_path = os.path.join(ASR_root, vid+".json") #转录结果保存的路径
    if os.path.exists(ASR_path):
        print("%s exists." % vid) 
        return

    #线程开始
    print("%s started with gpu %d." % (vid, gpu_id)) 

    video_path = os.path.join(video_root, vid+".mp4") #视频文件路径
    #whisper转录
    result = model[gpu_id].transcribe(video_path) 
    #保存转录结果
    with open(ASR_path, 'w') as f: 
        json.dump(result, f)

    #线程结束
    print("%s finished with gpu %d." % (vid, gpu_id)) 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=8, type=int)
    args = parser.parse_args()

    # 定义要使用的GPU数量
    num_gpus = args.gpus
    
    # 创建线程列表
    threads = []

    #每个gpu上部署一个whisper model
    model = [whisper.load_model("large").to(torch.device(f'cuda:{i}')) for i in range(num_gpus)]
    print('%d whisper loaded.' % (len(model)))

    #为每个gpu创建一个线程池
    pool = [ThreadPoolExecutor(max_workers=1) for i in range(num_gpus)]
    print('%d pool built.' % (len(pool)))

    #遍历数据集
    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    #random.shuffle(videos)
            
    #对每个视频创建处理线程
    for i, vid in enumerate(videos):
        gpu_id = i%num_gpus
        t = pool[gpu_id].submit(train_on_gpu, gpu_id, vid)
        threads.append(t)
            
    # 等待所有线程完成
    flag = True
    while flag:
        flag = False
        for t in threads:
            if not t.done():
                flag = True
    print('All subprocesses done.')