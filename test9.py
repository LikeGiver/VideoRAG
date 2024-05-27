from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'text': "这是一个第一视角拍摄视频中截取的几个关键帧，这是第一帧："},
    {'image': '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3631.png'},
    {'text': '这是第二帧'},
    {'image': '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3633.png'}, # Either a local path or an url
    {'text': '这是第三帧'},
    {'image': '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3635.png'},
    {'text': '请详细描述一下这个视频中人物的动作，发生的时间，出现的物品'}
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# # 2nd dialogue turn
# response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
# print(response)
# # <ref>击掌</ref><box>(536,509),(588,602)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('1.jpg')
# else:
#   print("no box")