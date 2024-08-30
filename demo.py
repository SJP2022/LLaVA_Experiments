'''
python demo.py --model-path llava-v1.6-vicuna-7b --load-4bit
'''
import argparse
import json
import os
import random
import time
import torch
import cv2

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from decord import VideoReader

class QuietTextStreamer(TextStreamer):
    def put(self, value):
        pass
    def on_finalized_text(self, text, stream_end):
        pass
    def end(self):
        pass

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, frame_idx):
    vr = VideoReader(video_path, ctx=cpu(0))
    start_frame = round(start_time*vr.get_avg_fps())
    end_frame = min(round(end_time*vr.get_avg_fps()), len(vr))
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(start_frame, end_frame, fps)]
    # sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    #print(start_time, end_time, frame_idx, len(vr))
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # Save frames as images
    # for i, frame in enumerate(spare_frames):
    #     cv2.imwrite(f'{args.output_dir}/frame_{i}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return spare_frames

video_root = "/share/common/VideoDatasets/ActivityNet/videos"
framecaps_root = "/share/test/shijiapeng/ActivityNet_annotations_24_8/ActivityNet_frames2Caption_llava1.6_7b"
#framecaps_root = "/share/test/shijiapeng/ActivityNet_annotations_24_8/ActivityNet_frames2Caption_exp"
shots_root = "/share/test/shijiapeng/ActivityNet_shots_final"

def get_duration(video_path):
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    return round(len(vr) / fps, 2), fps

def captioning(vid):
    #判断文件是否存在
    json_path = os.path.join(framecaps_root, "v_"+vid+".json")
    if os.path.exists(json_path):
        print("%s exists." % vid)
        return

    print("%s starts at %s" % (vid, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    framecaps = []

    video_path = os.path.join(video_root, vid+".mp4")
    shots_path = os.path.join(shots_root, vid+".json")
    duration, fps = get_duration(video_path)
    if os.path.exists(shots_path):
        with open(shots_path, 'r', encoding='utf-8') as f:
            shots = json.load(f) #list
    else:
        print("shots of %s don't exist. Can't summarize this video!!!" % vid)
        return
    
    cnt = len(shots)-1
    start_times = []
    end_times = []
    for idx in range(cnt):
        start_times.append(shots[idx])
        end_times.append(shots[idx+1])
    cap = cv2.VideoCapture(video_path)
    for start_time, end_time in zip(start_times, end_times):
        middle_sec = (start_time + end_time) // 2
        middle_frame_idx = int(middle_sec * fps)
        print(start_time, end_time, middle_frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
            
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            user_content = "Please provide a detailed description of the photo, focusing on the main subjects, their actions, the background scenes."
            conv = new_chat()
            inp = user_content

            image = Image.fromarray(frame)
            image_size = image.size
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                image = None
            
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = QuietTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True)

            outputs = tokenizer.decode(output_ids[0]).strip()
            caption = outputs.replace("<s>", "").replace("</s>", "").strip()
            print(caption)
            #framecaps.append(caption)
            framecaps.append({
                "id": idx,
                "start_time": start_time,
                "end_time": end_time,
                "captions": caption
            })
    cap.release()

    save_item = {
        "video_name": "v_"+vid,
        "fps": fps,
        "duration": duration,
        "captions": framecaps
    }
    
    with open(json_path, "w") as f:
        json.dump(save_item, f)

    print("%s finished at %s" % (vid, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

def main(args):
    # Model
    disable_torch_init()

    global model_name
    model_name = get_model_name_from_path(args.model_path)
    global tokenizer, model, image_processor, context_len 
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # Chat
    global new_chat
    def new_chat():
        conv = conv_templates[args.conv_mode].copy()
        return conv

    # Captioning
    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    random.shuffle(videos)
    
    #videos = ['WOUkPgHtt4E']
    #videos = ['2zVpWu1i5qM']
    #videos = ['0AjYz-s4Rek']
    #videos = ['sFKOnFMJF2Q']
    #videos = ['sWEbq5Ry63Q']
    #videos = ['37Q3so6ERxs']
    for vid in videos:
        captioning(vid)
        with open(os.path.join(framecaps_root, 'v_'+vid+'.json'), 'r', encoding='utf-8') as f:
            nota = json.load(f) #list
        for item in nota['captions']:
            if item['captions']=='':
                print("%s doesn't work!" % vid)
                break

    print("all finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
