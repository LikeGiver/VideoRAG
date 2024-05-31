from RAG.VectorBase import VectorStore
# from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import CNCLIP_Embedding
from RAG.utils import load_image

vector = VectorStore()

embedding = CNCLIP_Embedding()

#################### 4.1 混合模态检索尝试 ###################

# vector.upload_and_embed_files(embedding, document=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], 
#                               image_paths=['/home/ubuntu/data/user01/codes/VideoRAG/image_data/pokemon.jpeg', 
#                                            '/home/ubuntu/data/user01/codes/VideoRAG/image_data/d0ee81f16175c97770192fb691fdda8da1f4f349.png'])

# result = vector.query(text_query="pokemon", EmbeddingModel=embedding, k=5)

# print(result) # ['皮卡丘', '妙蛙种子', '小火龙', '杰尼龟', '/home/ubuntu/data/user01/codes/VideoRAG/image_data/pokemon.jpeg']


#################### 4.2 跨模态相似度探索 ###################
# query_feature = embedding.get_embedding(text="Pikachu")

# image_feature_1 = embedding.get_embedding(image=load_image('/home/ubuntu/data/user01/codes/VideoRAG/image_data/皮卡丘.jpeg'))

# similarity_1 = CNCLIP_Embedding.cosine_similarity(query_feature, image_feature_1)

# text_feature_1 = embedding.get_embedding(text='皮卡丘')

# similarity_2 = CNCLIP_Embedding.cosine_similarity(query_feature, text_feature_1)

# text_feature_2 = embedding.get_embedding(text='妙蛙种子')

# similarity_3 = CNCLIP_Embedding.cosine_similarity(query_feature, text_feature_2)

# image_feature_2 = embedding.get_embedding(image=load_image('/home/ubuntu/data/user01/codes/VideoRAG/image_data/小火龙.png'))

# similarity_4 = CNCLIP_Embedding.cosine_similarity(query_feature, image_feature_2)

# print(similarity_1, similarity_2, similarity_3, similarity_4) # 0.40600467 0.7155843 0.64107406 0.3458285 第一个比第四个大，第二个比第三个大，说明模态之间有可比性，跨模态之间无可比性


#################### 4.3 增加检索模态控制参数k后 ###################
vector.upload_and_embed_files(embedding, document=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], 
                              image_paths=['/home/ubuntu/data/user01/codes/VideoRAG/image_data/小火龙.png', 
                                           '/home/ubuntu/data/user01/codes/VideoRAG/image_data/皮卡丘.jpeg'])

# result = vector.query(text_query="小火龍", EmbeddingModel=embedding, text_k=3, image_k=1)

# print(result) # ['小火龙', '妙蛙种子', '皮卡丘', '/home/ubuntu/data/user01/codes/VideoRAG/image_data/小火龙.png']

result, sim = vector.query(image_path_query='/home/ubuntu/data/user01/codes/VideoRAG/image_data/小火龙2.png', EmbeddingModel=embedding, text_k=3, image_k=1)

print(result, '\n', sim)