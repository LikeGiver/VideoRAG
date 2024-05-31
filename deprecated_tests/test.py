from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14", cache_dir="/home/ubuntu/data/user01/codes/VideoRAG/chinese-clip-vit-large-patch14", local_files_only=True)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14", cache_dir="/home/ubuntu/data/user01/codes/VideoRAG/chinese-clip-vit-large-patch14", local_files_only=True)

url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

# compute image feature
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute text features
inputs = processor(text=texts, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute image-text similarity scores
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

print(logits_per_image)

probs = logits_per_image.softmax(dim=1)  # probs: [[0.0066, 0.0211, 0.0031, 0.9692]]

print(probs)
