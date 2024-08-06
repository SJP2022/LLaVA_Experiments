"""
python vicuna.py --model-path lmsys/vicuna-13b-v1.5 --num-gpus 2 --max-gpu-memory 24GiB
"""
import argparse
import os, re, sys, json, math, time
import random
from typing import Iterable, Optional, Dict
import warnings
import abc
import gc
import psutil
import torch
from decord import VideoReader
from projects.FastChat.fastchat.model.model_adapter import add_model_args
from projects.FastChat.fastchat.modules.awq import AWQConfig
from projects.FastChat.fastchat.modules.exllama import ExllamaConfig
from projects.FastChat.fastchat.modules.xfastertransformer import XftConfig
from projects.FastChat.fastchat.modules.gptq import GptqConfig
from projects.FastChat.fastchat.utils import str_to_torch_dtype
from projects.FastChat.fastchat.conversation import get_conv_template, SeparatorStyle
from projects.FastChat.fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length
from projects.FastChat.fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)

#video_root = "demo_videos"
video_root = "videos"
shots_root = "/share/test/shijiapeng/ActivityNet_shots_HOG_filter"
framecaps_root = "framecaps"
videocaps_root = "videocaps"
ASR_root = "ASR"
#notations_root = "demo_notations/test"
notations_root = "/share/test/shijiapeng/ActivityNet_annotations_new/ActivityNet_annotations_vicuna"
densecap_root = "densecap"
labels_path = "vlabels/labels.json"
ASR_train = os.listdir(os.path.join(ASR_root, "train"))
ASR_val = os.listdir(os.path.join(ASR_root, "val"))
ASR_test = os.listdir(os.path.join(ASR_root, "test"))
with open(os.path.join(densecap_root, "train.json"), 'r', encoding='utf-8') as f:
    densecap_train = json.load(f)
with open(os.path.join(densecap_root, "val_1.json"), 'r', encoding='utf-8') as f:
    densecap_val = json.load(f)
with open(os.path.join(densecap_root, "val_2.json"), 'r', encoding='utf-8') as f:
    densecap_val2 = json.load(f)
    densecap_val.update(densecap_val2)
with open(labels_path, 'r', encoding='utf-8') as f:
    labels_data = json.load(f)
labels_list = labels_data["database"]

def get_duration(video_path):
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    return round(len(vr) / fps, 2), fps

def summary(vid):
    #判断文件是否存在
    json_path = os.path.join(notations_root, vid+".json")
    if os.path.exists(json_path):
        print("%s exists." % vid)
        return

    notations = []

    video_path = os.path.join(video_root, vid+".mp4")
    shots_path = os.path.join(shots_root, vid+".json")
    framecaps_path = os.path.join(framecaps_root, "v_"+vid+".json")
    videocaps_path = os.path.join(videocaps_root, "v_"+vid+".json")
    #ASR path & densecap & label
    label = ""
    labels = labels_list[vid]["annotations"]
    if len(labels)>0:
        #label = labels[0]["label"]
        label_set = set()
        for label_item in labels:
            label_set.add(label_item["label"])
        for label_name in label_set:
            label += (label_name+". ")
        label = label[:-1]
    #densecap = ""
    densecap_list = []
    if vid+".json" in ASR_train:
        ASR_path = os.path.join(os.path.join(ASR_root, "train"), vid+".json")
        #densecap_list = densecap_train["v_"+vid]["sentences"]
        if "v_"+vid in densecap_train:
            densecap_list = densecap_train["v_"+vid]
        else:
            print("densecap of %s don't exist." % vid)
    elif vid+".json" in ASR_val:
        ASR_path = os.path.join(os.path.join(ASR_root, "val"), vid+".json")
        #densecap_list = densecap_val["v_"+vid]["sentences"]
        if "v_"+vid in densecap_val:
            densecap_list = densecap_val["v_"+vid]
        else:
            print("densecap of %s don't exist." % vid)
    else:
        ASR_path = os.path.join(os.path.join(ASR_root, "test"), vid+".json")
        print("densecap of %s don't exist." % vid)
    #for cap in densecap_list:
    #    densecap += cap
    duration, fps = get_duration(video_path)
    
    if os.path.exists(shots_path):
        with open(shots_path, 'r', encoding='utf-8') as f:
            shots = json.load(f) #list
    else:
        print("shots of %s don't exist. Can't summarize this video!!!" % vid)
        return
    
    if os.path.exists(ASR_path):
        with open(ASR_path, 'r', encoding='utf-8') as f:
            ASR_data = json.load(f) 
        ASR = ASR_data["segments"] #list
    else:
        print("ASRs of %s don't exist. Can't summarize this video!!!" % vid)
        return
    
    if os.path.exists(framecaps_path):
        with open(framecaps_path, 'r', encoding='utf-8') as f:
            framecaps_data = json.load(f)
        framecaps = framecaps_data["captions"] #list
    else:
        print("framecaps of %s don't exist. Can't summarize this video!!!" % vid)
        return

    if os.path.exists(videocaps_path):
        with open(videocaps_path, 'r', encoding='utf-8') as f:
            videocaps_data = json.load(f)
        videocaps = videocaps_data["omnivl_caption"] #string
    else:
        print("videocaps of %s don't exist." % vid)
        videocaps = ""

    if len(densecap_list)>0:
        print("%s starts and uses densecap: %s" % (vid, densecap_list))
    elif label!="":
        print("%s starts and uses label: %s" % (vid, label))
    elif videocaps!="":
        print("%s starts and uses videocap: %s" % (vid, videocaps))
    else:
        print("%s starts and uses no caps." % vid)

    #print("%s starts." % vid)

    cnt = len(shots)
    start_times = []
    end_times = []
    for idx in range(cnt):
        start_times.append((shots[idx]-1)/fps)
        if idx < cnt-1:
            end_times.append((shots[idx+1]-1)/fps)
        else:
            end_times.append(duration)

    for idx in range(cnt):
        start_time = start_times[idx]
        end_time = end_times[idx]
        #settings
        user_content = "I will provide you with information about a video segment that has been cut from a longer video, including automatically recognized speech with timestamps, captions for frames of the segment with timestamps, and the summary for the longer video. Write a summary for this video segment. Write only short sentences. Describe only one action per sentence. Keep only actions that happen in the present time. Note that ASR and the summary for the longer video may not exist.\n" 
        #Write a summary for this video segment, using only short sentences to describe one action per sentence, focusing only on actions that are happening in the present time.
        #ASR
        user_content += "Here is the automatically recognized speech: <ASR with timestamps in the format \"n\"s: \"ASR\">\n"
        flag = False #是否有ASR
        for seg in ASR:
            if seg['start']>=start_time and seg['start']<=end_time:
                flag = True
                user_content += ("%ss: %s\n" % (seg['start'], seg['text']))
            elif seg['start']>end_time:
                break
        if not flag:
            user_content += "No ASR in this segment.\n"
        #framecaps
        user_content += "Here are the captions for frames of the segment: <captions with timestamps in the format \"n\"s: \"caption\">\n"
        st = math.ceil(start_time)
        ed = math.floor(end_time)
        for sec in range(st, ed+1):
            if sec<len(framecaps):
                user_content += ("%ss: %s\n" % (sec, framecaps[sec]))
        #videocaps
        user_content += "Here is the summary for the longer video where this segment has been cut from:\n"
        #'''
        if len(densecap_list)>0:
            densecap = ""
            for sen_id, timestamp in enumerate(densecap_list["timestamps"]):
                if not (end_time<=timestamp[0] or start_time>=timestamp[1]):
                    densecap += densecap_list["sentences"][sen_id]
            user_content += (densecap.strip()+"\n")
        elif label!="":
            user_content += (label+"\n")
        elif videocaps!="":
            user_content += (videocaps+"\n")
        else:
            user_content += "No summary for the longer video.\n"
        #'''
        #user_content += (videocaps+"\n")
        #user_content += (densecap+"\n")
        #summary formats
        user_content += ("The length of the whole video is %ss and this segment is from %ss to %ss. " % (duration, start_time, end_time))
        user_content += "Now please summarize the current segment based on its ASR and captions for frames, combining the context of the summary for the longer video where this segment has been cut from. Please exclude unrelated words, such as \"I hope that helps!\" or \"Sure, I'd be happy to help!\" from your summary. And no need to provide any timestamp or person name in your summary.\n"
        #And please make sure that the ASR, the captions for frames of the segment and the caption for the whole long video where this segment has been cut from are all taken into consideration.
        conv = new_chat()
        inp = user_content
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        #print(model.device)
        output_stream = generate_stream_func(
            model,
            tokenizer,
            gen_params,
            device,
            context_len=context_len,
            stream_interval=max_new_tokens, #调节刷新频率
            judge_sent_end=judge_sent_end,
        )
        #t = time.time()
        #print(max_new_tokens)
        for outputs in output_stream:
            #print(outputs)
            output_text = outputs["text"]
        #outputs = output_text.strip()
        #duration = time.time() - t

        #print(results)
        #result_content = output_text.split('\n\n', 2)[1].strip().replace('\n', ' ')
        result_content = output_text.strip().replace('\n', ' ')
        result_sentences = [sentence.strip() for sentence in result_content.split('. ') if sentence.strip()]
        result_sentences[len(result_sentences)-1] = result_sentences[len(result_sentences)-1][:-1]
        
        start_frame = shots[idx]
        if idx < cnt-1:
            end_frame = shots[idx+1]-1
        else:
            end_frame = round(duration*fps)

        notations.append({
            "id": idx,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "summary": result_sentences
        })

        #print(conv.dict())
        print("SYSTEM: %s" % conv_system_msg)
        print("USER: %s" % inp)
        print("ASSISTANT: %s" % output_text)
        print("\n==================================\n")

    save_item = {
        "video_name": vid,
        "fps": fps,
        "duration": duration,
        "notations": notations
    }
    
    with open(json_path, "w") as f:
        json.dump(save_item, f)

    print("%s finished." % vid)

def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    #chatio = SimpleChatIO(args.multiline)
    
    # parameters
    global model_path, device, num_gpus, max_gpu_memory, dtype, load_8bit, cpu_offloading, conv_template, conv_system_msg, temperature, repetition_penalty, max_new_tokens, exllama_config, xft_config, gptq_config, awq_config, revision, judge_sent_end, debug, history
    model_path = args.model_path
    device = args.device
    num_gpus = args.num_gpus
    max_gpu_memory = args.max_gpu_memory
    dtype = str_to_torch_dtype(args.dtype)
    load_8bit = args.load_8bit
    cpu_offloading = args.cpu_offloading
    conv_template = args.conv_template
    conv_system_msg = args.conv_system_msg
    temperature = args.temperature
    repetition_penalty = args.repetition_penalty
    max_new_tokens = args.max_new_tokens
    exllama_config = None
    xft_config = None
    gptq_config = GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                )
    awq_config = AWQConfig(
                    ckpt=args.awq_ckpt or args.model_path,
                    wbits=args.awq_wbits,
                    groupsize=args.awq_groupsize,
                )
    revision = args.revision
    judge_sent_end = args.judge_sent_end
    debug = args.debug
    history = not args.no_history

    # Model
    global model, tokenizer, generate_stream_func
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=revision,
        debug=debug,
    )
    generate_stream_func = get_generate_stream_function(model, model_path)
    print("Model loaded.")
    
    # Set context length
    global context_len
    context_len = get_context_length(model.config)

    # Chat
    global new_chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    # Set system_content
    system_content = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    conv_system_msg = system_content

    # Summary
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
    #videos = ['ZL7xefcDWYc']
    for vid in videos:
        summary(vid)

    print("all finished!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)