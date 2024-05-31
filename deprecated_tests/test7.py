import os
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.manual_seed(1234)

# 定义文件夹路径
folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda'

# 获取文件夹内所有图片文件的路径
image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpeg', '.jpg'))]

vector = VectorStore(auto_load=True, load_path='/home/ubuntu/data/user01/codes/VideoRAG/image_data/1dcc108c-8bd4-42ad-b2c5-03662be62eda')
embedding = CNCLIP_Embedding()

# 上传和嵌入图片文件
# 仅在初始加载时使用
# vector.upload_and_embed_files(embedding, 
#                               image_paths=image_paths, 
#                               save_path='/home/ubuntu/data/user01/codes/VideoRAG/image_data/1dcc108c-8bd4-42ad-b2c5-03662be62eda')

def query_images(vector, embedding, text_query, text_k=0, image_k=3):
    result, sim = vector.query(text_query=text_query, EmbeddingModel=embedding, text_k=text_k, image_k=image_k)
    print(f"Query: {text_query}")
    print("Results:", result)
    print("Similarities:", sim)
    return result, sim

# 示例查询
query_text = 'Did I leave the car door open?'
result, sim = query_images(vector, embedding, query_text)

# 初始化 Qwen-VL 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/data/user01/codes/VideoRAG/Qwen-VL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/data/user01/codes/VideoRAG/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

def ask_question_with_images(model, tokenizer, image_paths, question):
    query = [{'image': img_path} for img_path in image_paths]
    query.append({'text': question})
    inputs = tokenizer.from_list_format(query)
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print(response)
    return response

# 使用检索到的图片地址列表和问题进行问答
question = 'What tool did I use to tighten the car handle?'
response = ask_question_with_images(model, tokenizer, result, question)

# 循环查询示例
while True:
    query_text = input("Enter your query (or 'exit' to quit): ")
    if query_text.lower() == 'exit':
        break
    result, sim = query_images(vector, embedding, query_text)
    question = input("Enter your question based on the retrieved images: ")
    response = ask_question_with_images(model, tokenizer, result, question)