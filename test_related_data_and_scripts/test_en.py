import json
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding

torch.manual_seed(1234)

embedding = CNCLIP_Embedding()

def load_vector_store(load_path):
    return VectorStore(auto_load=True, load_path=load_path)

def query_images(vector, embedding, text_query, text_k=0, image_k=3):
    result, sim = vector.query(text_query=text_query, EmbeddingModel=embedding, text_k=text_k, image_k=image_k)
    print(f"Query: {text_query}")
    print("Results:", result)
    print("Similarities:", sim)
    return result, sim

def ask_question_with_images(model, tokenizer, image_paths, question):
    query = [{'text': "These are three video frames retrieved from a first-person video. Please answer the question based on the content in the image."}]
    query.extend([{'image': image_path} for image_path in image_paths])
    query.append({'text': question})
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
            result, _ = query_images(vector, embedding, question)
            if not result:
                print(f"No images found for question: {question}")
                continue
            response = ask_question_with_images(self.model, self.tokenizer, result, question)
            
            bleu_score = sentence_bleu([answer.split()], response.split())
            meteor = nltk.translate.meteor_score.single_meteor_score(set(answer.split()), set(response.split()))
            # rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            # rouge_scores = rouge.score(answer, response)
            # 创建ROUGE计算器，只计算ROUGE-L
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            # 计算ROUGE-L分数
            rouge_scores = rouge.score(answer, response)
            # 提取ROUGE-L的F-measure
            rougeL_fmeasure = rouge_scores['rougeL'].fmeasure
            
            scores['bleu'].append(bleu_score)
            scores['meteor'].append(meteor)
            scores['rougeL_fmeasure'].append(rougeL_fmeasure)
            
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Response: {response}")
            print(f"BLEU: {bleu_score}")
            print(f"METEOR: {meteor}")
            print(f"ROUGE-L: {rouge_scores}")
            print('-' * 40)
        
        avg_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}
        return avg_scores

# 示例使用
evaluator = VideoQAEvaluator(
    json_path='/home/ubuntu/data/user01/codes/ego4d/data/unified/annotations.QaEgo4D_test.json',
    model_path="Qwen/Qwen-VL-Chat",
    tokenizer_path="Qwen/Qwen-VL-Chat"
)

video_uid = '2ac65951-49b0-4629-981e-edc34b8cdb0f'
vector_store_path = '/home/ubuntu/data/user01/codes/VideoRAG/image_data/' + video_uid
average_scores = evaluator.evaluate(video_uid, vector_store_path)
print("Average Scores:", average_scores)