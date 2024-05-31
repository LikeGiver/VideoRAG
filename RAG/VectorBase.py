#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   VectorBase.py
@Time    :   2024/04/09 22:17:00
@Author  :   不要葱姜蒜, LikeGiver
@Version :   1.1
@Desc    :   None
'''

import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings
from RAG.utils import load_image
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

AUTOSAVE_PATH = 'storage'
AUTOLOAD_PATH = 'storage'

class VectorStore:
    def __init__(self, auto_load: bool = False, load_path: str = AUTOLOAD_PATH) -> None:
        self.vectors = {}
        self.video_frame_ids = [] # 每个VectorStore仅限一条视频，除非你想加在之前的视频帧之后
        if not auto_load:
            self._load_vector(path=load_path)
        else:
            self._load_vector(path=AUTOLOAD_PATH)
            
    def _generate_unique_id(self, source_string: str) -> str:
        """
        虽然时字符串到字符串的映射，但应该更短
        """
        import hashlib
        hash_value = hashlib.sha256(source_string.encode('utf-8')).hexdigest()
        return hash_value
    
    def upload_and_embed_files(self, EmbeddingModel: BaseEmbeddings = None, document: List[str] = [], image_paths: List[str] = [], frame_paths: List[str] = [], auto_persist: bool = True, save_path: str = AUTOSAVE_PATH):
        if EmbeddingModel is None:
            raise ValueError("An embedding model must be provided.")
        
        for doc in tqdm(document, desc="Calculating embeddings for text"):
            unique_id = self._generate_unique_id(doc)  # 为每个文本生成唯一ID
            vector = EmbeddingModel.get_embedding(text=doc)
            self.vectors[unique_id] = {'vector': vector.tolist(), 'type': 'text', 'content': doc}

        for img_path in tqdm(image_paths, desc="Calculating embeddings for images"):
            img = load_image(img_path)
            unique_id = self._generate_unique_id(img_path)  # 为每个图像路径生成唯一ID
            vector = EmbeddingModel.get_embedding(image=img)
            self.vectors[unique_id] = {'vector': vector.tolist(), 'type': 'image', 'path': img_path}
        
        for frame_path in tqdm(frame_paths, desc="Calculating embeddings for frames"):
            frame_img = load_image(frame_path)
            unique_id = self._generate_unique_id(frame_path)
            vector = EmbeddingModel.get_embedding(image=frame_img)
            self.vectors[unique_id] = {'vector': vector.tolist(), 'type': 'frame', 'path': frame_path}
            self.video_frame_ids.append(unique_id)
            
        if auto_persist:
            self._persist(path=save_path)

    def _persist(self, path: str = AUTOSAVE_PATH):
        """
        将当前的向量、文本内容和图像路径信息持久化到指定路径。
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        # 准备要保存的数据。如果向量是NumPy数组，则需要转换为列表
        data_to_save = {}
        for unique_id, info in self.vectors.items():
            vector = info['vector']
            if isinstance(vector, np.ndarray):
                # 如果向量是NumPy数组，转换为列表
                vector = vector.tolist()
            # 更新向量信息，确保其可以被JSON序列化
            data_to_save[unique_id] = {**info, 'vector': vector}
            
        # 保存到文件
        vectors_path = os.path.join(path, "vectors.json")
        with open(vectors_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        
        video_frames_ids_path = os.path.join(path, "video_frames_ids.json")   
        with open(video_frames_ids_path, 'w') as f:
            json.dump(self.video_frame_ids, f)
            
        print(f"Data persisted at {vectors_path}")

    def _load_vector(self, path: str = None):
        """
        从给定路径加载向量、文档和图像路径。
        """
        if path is None:
            raise ValueError("path should not be None!")
        vectors_path = os.path.join(path, "vectors.json")
        if os.path.exists(vectors_path):
            with open(vectors_path, 'r', encoding='utf-8') as f:
                # 加载向量及其相关信息（假设保存格式符合预期的字典结构）
                self.vectors = json.load(f) 
        else:
            print(f"No vectors found at {vectors_path}. Starting with an empty store.")
        
        video_frames_ids_path = os.path.join(path, "video_frames_ids.json")
        if os.path.exists(video_frames_ids_path):
           with open(video_frames_ids_path, 'r') as f:
                self.video_frame_ids = json.load(f) 
        else:
            print(f"No vectors found at {video_frames_ids_path}. Starting without video frames.")
            
    def description_generate(self, EmbeddingModel: BaseEmbeddings, saved_path: str = AUTOSAVE_PATH):
        if self.video_frame_ids == []:
            print("Please upload video frames first!")
            return
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        system_prompt: str = "这些图片是从第一视角拍摄的视频中取出的视频帧：\n"
        previous_text = ""
        
        for i in tqdm(range(0, len(self.video_frame_ids), 5), desc="Generating descriptions"):
            window_frames = self.video_frame_ids[i:i+5]
            # queries = [{'text': "你是一个事件的记录员，这是事件发生的历史背景：" + previous_text}, {'text': system_prompt}]
            queries = [{'text': system_prompt}]
            queries.extend([{'image': self.vectors[frame_id]['path']} for frame_id in window_frames])
            queries.append({'text': '请推理一下在这个视频中发生的事件，并描述一下出现的物品，越详细越好。'})

            query = tokenizer.from_list_format(queries)
            response, _ = model.chat(tokenizer, query=query, history=None)
            description_embedding = EmbeddingModel.get_embedding(text=response).tolist()
            
            for frame_id in window_frames:
                self.vectors[frame_id]['description'] = response
                self.vectors[frame_id]['description_embedding'] = description_embedding
            
            previous_text = response
            print(f"Description for frames {window_frames}:\n{response}\n")
        
        self._persist(path=saved_path)
    
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)
    
    def query(self, text_query: str = None, image_path_query: str = None, EmbeddingModel = None, text_k: int = 1, image_k: int = 0, frame_k: int = 0) -> Tuple[List[str], List[float]]:
        if text_query is None and image_path_query is None:
            raise ValueError("You have to specify either text_query or image_path_query. Both cannot be none.")
        if EmbeddingModel is None:
            raise ValueError("An embedding model must be provided.")
        
        # 根据查询类型获取查询向量
        if text_query:
            query_vector = EmbeddingModel.get_embedding(text=text_query).tolist()
        elif image_path_query:
            image = load_image(image_path_query)
            query_vector = EmbeddingModel.get_embedding(image=image).tolist()

        # 计算所有向量与查询向量的相似度
        similarities = []
        for unique_id, vector_info in self.vectors.items():
            vector = vector_info['vector']
            similarity = self.get_similarity(query_vector, vector)
            if vector_info['type'] == 'frame':
                description_similarity = self.get_similarity(query_vector, vector_info['description_embedding'])
                similarity = (similarity + description_similarity) / 2
            similarities.append((unique_id, similarity, vector_info['type']))


        # 分别对文本和图像结果应用不同的top_k值，并记录相似度
        text_results_with_sim = sorted(
            [(unique_id, sim) for unique_id, sim, vector_type in similarities if vector_type == 'text'],
            key=lambda x: x[1], reverse=True
        )[:text_k]
        
        image_results_with_sim = sorted(
            [(unique_id, sim) for unique_id, sim, vector_type in similarities if vector_type == 'image'],
            key=lambda x: x[1], reverse=True
        )[:image_k]
        
        frame_results_with_sim = sorted(
            [(unique_id, sim) for unique_id, sim, vector_type in similarities if vector_type == 'frame'],
            key=lambda x: x[1], reverse=True
        )[:frame_k]

        # 提取结果和相似度
        results, sims = [], []
        for unique_id, sim in text_results_with_sim + image_results_with_sim + frame_results_with_sim:
            if self.vectors[unique_id]['type'] == 'text':
                results.append(self.vectors[unique_id]['content'])
            elif self.vectors[unique_id]['type'] == 'image':
                results.append(self.vectors[unique_id]['path'])
            elif self.vectors[unique_id]['type'] == 'frame':
                results.append((self.vectors[unique_id]['path'], self.vectors[unique_id]['description']))
            sims.append(sim)

        return results, sims

