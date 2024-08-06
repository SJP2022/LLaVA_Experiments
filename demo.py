'''
python demo.py --model-path llava-v1.6-vicuna-13b --load-4bit
'''
import argparse
import json
import os
import random
import time
import torch

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

frames_root = "/share/test/chengfeng/ActivityNet_frames"
video_root = "/share/common/VideoDatasets/ActivityNet/videos"
framecaps_root = "/share/test/shijiapeng/ActivityNet_annotations_24_8/ActivityNet_frames2Caption_llava1.6"

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
    frames_path = os.path.join(frames_root, vid)
    duration, fps = get_duration(video_path)
    if not os.path.exists(frames_path):
        print("frames of %s don't exist. Can't captioning!!!" % vid)
        return
    
    frames = sorted(os.listdir(frames_path))
    for img in frames:
        img_path = os.path.join(frames_path, img)
        #settings
        #user_content = "Please write a caption for this photo. Write only short sentences. Describe only one action per sentence." 
        #user_content = "Please write a caption for this photo." 
        #user_content = "a photo of"
        user_content = "Describe this photo in detail. Include details like object counts, position of the objects, relative position between the objects. Always answer as if you are directly looking at the image."
        conv = new_chat()
        inp = user_content

        image = load_image(img_path)
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
        #print(caption)
        framecaps.append(caption)

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
    
    #videos = ['00SfeRtiM2o']
    #videos = ['2zVpWu1i5qM']
    #videos = ['0AjYz-s4Rek']
    #videos = ['sFKOnFMJF2Q']
    #videos = ['sWEbq5Ry63Q']
    #videos = ['37Q3so6ERxs']
    for vid in videos:
        captioning(vid)

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
