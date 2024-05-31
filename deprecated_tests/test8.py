import os
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

torch.manual_seed(1234)

# # 定义文件夹路径
# folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda'

# # 获取文件夹内所有图片文件的路径
# image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpeg', '.jpg'))]

embedding = CNCLIP_Embedding()

def load_vector_store(load_path):
    return VectorStore(auto_load=True, load_path=load_path)

def query_images(vector, embedding, text_query, text_k=0, image_k=3):
    result, sim = vector.query(text_query=text_query, EmbeddingModel=embedding, text_k=text_k, image_k=image_k)
    print(f"Query: {text_query}")
    print("Results:", result)
    print("Similarities:", sim)
    return result, sim

# 示例查询
vector = load_vector_store('/home/ubuntu/data/user01/codes/VideoRAG/image_data/1dcc108c-8bd4-42ad-b2c5-03662be62eda')
query_text = 'Did I leave the car door open?'
result, sim = query_images(vector, embedding, query_text)

# 初始化 Qwen-VL 模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/data/user01/codes/VideoRAG/Qwen-VL", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/data/user01/codes/VideoRAG/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def ask_question_with_images(model, tokenizer, image_paths, question):
    # query = [{'image': img_path} for img_path in image_paths]
    query = [{'image': image_paths[0]}]
    query.append({'text': question})
    formated_query = tokenizer.from_list_format(query)
    # inputs = tokenizer(formated_query, return_tensors='pt')
    # inputs = inputs.to(model.device)
    # pred = model.generate(**inputs)
    # response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
    response, history = model.chat(tokenizer, query=formated_query, history=None)
    print(response)
    return response

# 使用检索到的图片地址列表和问题进行问答
question = 'What tool did I use to tighten the car handle?'
response = ask_question_with_images(model, tokenizer, result, question)

# 循环查询示例
while True:
    load_path = input("Enter the vector store load path (or 'exit' to quit): ")
    if load_path.lower() == 'exit':
        break
    vector = load_vector_store(load_path)
    query_text = input("Enter your query: ")
    result, sim = query_images(vector, embedding, query_text)
    question = input("Enter your question based on the retrieved images: ")
    response = ask_question_with_images(model, tokenizer, result, question)