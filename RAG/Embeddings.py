#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/02/10 21:55:39
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

CN_CLIP_CACHE_DIR = "/home/ubuntu/data/user01/codes/VideoRAG/chinese-clip-vit-large-patch14"

class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

# class OpenAIEmbedding(BaseEmbeddings):
#     """
#     class for OpenAI embeddings
#     """
#     def __init__(self, path: str = '', is_api: bool = True) -> None:
#         super().__init__(path, is_api)
#         if self.is_api:
#             from openai import OpenAI
#             self.client = OpenAI()
#             self.client.api_key = os.getenv("OPENAI_API_KEY")
#             self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
#     def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
#         if self.is_api:
#             text = text.replace("\n", " ")
#             return self.client.embeddings.create(input=[text], model=model).data[0].embedding
#         else:
#             raise NotImplementedError

class CNCLIP_Embedding(BaseEmbeddings):
    def __init__(self, path: str = '', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model, self._processor = self._load_model()
        
    def get_embedding(self, text: str = None, image: Image = None) -> List[float]:
        if text is None and image is None:
            raise ValueError("You have to specify either text or image. Both cannot be none.")
        if text is not None:
            inputs = self._processor(text=[text], padding=True, return_tensors="pt")
            return self._model.get_text_features(**inputs)[0].detach().numpy()
        if image is not None:
            inputs = self._processor(images=image, return_tensors="pt")
            return self._model.get_image_features(**inputs)[0].detach().numpy()
            
    def _load_model(self):
        from transformers import ChineseCLIPProcessor, ChineseCLIPModel
        model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14", cache_dir=CN_CLIP_CACHE_DIR, local_files_only=True)
        processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14", cache_dir=CN_CLIP_CACHE_DIR, local_files_only=True)
        return model, processor
    
# class JinaEmbedding(BaseEmbeddings):
#     """
#     class for Jina embeddings
#     """
#     def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
#         super().__init__(path, is_api)
#         self._model = self.load_model()
        
#     def get_embedding(self, text: str) -> List[float]:
#         return self._model.encode([text])[0].tolist()
    
#     def load_model(self):
#         import torch
#         from transformers import AutoModel
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         else:
#             device = torch.device("cpu")
#         model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
#         return model

class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) 
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
        model="embedding-2",
        input=text,
        )
        return response.data[0].embedding