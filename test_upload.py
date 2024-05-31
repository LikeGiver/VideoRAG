# nohup /home/ubuntu/data/user01/anaconda3/envs/video_rag/bin/python /home/ubuntu/data/user01/codes/VideoRAG/test_upload.py > output2.txt 2>&1 &
import os
import re
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# # 定义文件夹路径
# folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda'
# folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/2bd5c29e-ef5c-451c-aa2e-961d4257de9b'
folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/ee401f80-7732-4f67-a9bb-0c1e58b40b84'


# 获取文件夹内所有图片文件的路径
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpeg', '.jpg'))]

# 定义一个函数从字符串中提取数字
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# 按文件名中的数字排序
sorted_image_files = sorted(image_files, key=extract_number)

# 获取排序后文件的完整路径
frame_paths = [os.path.join(folder_path, file) for file in sorted_image_files]
    
    
save_path = '/home/ubuntu/data/user01/codes/VideoRAG/image_data/ee401f80-7732-4f67-a9bb-0c1e58b40b84'
vector = VectorStore(load_path=save_path)
embedding = CNCLIP_Embedding()

# 上传和嵌入图片文件
vector.upload_and_embed_files(embedding, 
                              frame_paths=frame_paths, 
                              save_path=save_path)

vector.description_generate(EmbeddingModel=embedding,
                            saved_path=save_path)

# 查询相似的图片
question='我洗过青椒吗？'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

query = []
query.append({'text': f"请你帮忙推理一下，为了回答这个问题：{question}，我们需要在一个第一视角拍摄的视频内找到什么样的图片？涉及哪些动作，物品与场景？"})
formated_query = tokenizer.from_list_format(query)
response, history = model.chat(tokenizer, query=formated_query, history=None)

print(response,'\n')

result, sim = vector.query(text_query=response, EmbeddingModel=embedding, text_k=0, image_k=0, frame_k=3)

print(result, '\n', sim)

# ['/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3179.png', '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3180.png', '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3181.png'] 
#  [0.3936922300736449, 0.39045389177953294, 0.38463209139615795]
