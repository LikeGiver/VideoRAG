import json
import os
import jieba  # 用于中文分词
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding

torch.manual_seed(1234)

embedding = CNCLIP_Embedding()

def load_vector_store(load_path):
    return VectorStore(load_path=load_path)

def query_images(vector, embedding, text_query, text_k=0, image_k=0, frame_k=3):
    result, sim = vector.query(text_query=text_query, EmbeddingModel=embedding, text_k=text_k, image_k=image_k, frame_k=frame_k)
    print(f"Query: {text_query}")
    print("Results:", result)
    print("Similarities:", sim)
    return result, sim

def ask_question_with_images(model, tokenizer, retrieved_results, question):
    query = [{'text': "你是一个根据图片内容回答对应问题的机器人，下面是几张第一视角拍摄的图片和图片拍摄的背景，几张图片来自同一个视频："}]
    for result in retrieved_results:
        query.append({'text' : "该图片拍摄的背景是:" + result[1] + "\n 它对应的图片如下："})
        query.append({'image': result[0]})
    query.append({'text': "\n请根据上方任意一张图片的信息回答如下问题:\n" + question})
    formated_query = tokenizer.from_list_format(query)
    response, history = model.chat(tokenizer, query=formated_query, history=None)
    print(response)
    return response

class VideoQAEvaluator:
    def __init__(self, json_path, model_path, tokenizer_path):
        with open(json_path, 'r') as file:
            self.data = json.load(file)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    
    def get_qa_pairs(self, video_uid):
        return [(item['question'], item['answer']) for item in self.data if item['video_uid'] == video_uid]
    
    def evaluate(self, video_uid, vector_store_path):
        vector = load_vector_store(vector_store_path)
        qa_pairs = self.get_qa_pairs(video_uid)
        scores = {'bleu': [], 'meteor': [], 'rougeL_fmeasure': []}
        
        for question, answer in qa_pairs:
            query = []
            query.append({'text': f"请你帮忙推理一下，为了回答这个问题：{question}，我们需要在一个第一视角拍摄的视频内找到什么样的图片？涉及哪些动作，物品与场景？"})
            formated_query = self.tokenizer.from_list_format(query)
            response, history = self.model.chat(self.tokenizer, query=formated_query, history=None)
            result, _ = query_images(vector, embedding, response)
            if not result:
                print(f"No images found for question: {question}")
                continue
            response = ask_question_with_images(self.model, self.tokenizer, result, question)
            
            # 分词处理
            def tokenize(text):
                return ' '.join(jieba.cut(text))
            # 使用jieba进行中文分词
            answer_tokens = tokenize(answer)
            response_tokens = tokenize(response)
            
            # 计算BLEU分数
            smoothie = nltk.translate.bleu_score.SmoothingFunction().method4
            bleu_score = sentence_bleu([answer_tokens.split()], response_tokens.split(), smoothing_function=smoothie)
            
            # 计算METEOR分数，nltk的meteor_score期望输入为字符串而不是分词后的列表
            # meteor = meteor_score([answer], response)
            # meteor = nltk.translate.meteor_score.single_meteor_score(set(list(jieba.cut(answer))), set(jieba.cut(response)))
            meteor = single_meteor_score(set(answer), set(response))
            
            # 计算ROUGE-L分数
            rouge = Rouge()
            rouge_scores = rouge.get_scores([response_tokens], [answer_tokens])
            rougeL_fmeasure = rouge_scores[0]['rouge-l']['f']
            
            scores['bleu'].append(bleu_score)
            scores['meteor'].append(meteor)
            scores['rougeL_fmeasure'].append(rougeL_fmeasure)
            
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Response: {response}")
            print(f"BLEU: {bleu_score}")
            print(f"METEOR: {meteor}")
            print(f"ROUGE-L: {rougeL_fmeasure}")
            print('-' * 40)
        
        avg_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}
        return avg_scores

# 示例使用
evaluator = VideoQAEvaluator(
    json_path='/home/ubuntu/data/user01/codes/VideoRAG/translated_data.json',
    model_path="Qwen/Qwen-VL-Chat",
    tokenizer_path="Qwen/Qwen-VL-Chat"
)

video_uid = 'ee401f80-7732-4f67-a9bb-0c1e58b40b84'
vector_store_path = '/home/ubuntu/data/user01/codes/VideoRAG/image_data/' + video_uid
average_scores = evaluator.evaluate(video_uid, vector_store_path)
print("Average Scores:", average_scores)