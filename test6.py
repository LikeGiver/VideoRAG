import os
from RAG.VectorBase import VectorStore
from RAG.Embeddings import CNCLIP_Embedding

# # 定义文件夹路径
# folder_path = '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda'

# # 获取文件夹内所有图片文件的路径
# image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpeg', '.jpg'))]

vector = VectorStore(auto_load=True, load_path='/home/ubuntu/data/user01/codes/VideoRAG/image_data/1dcc108c-8bd4-42ad-b2c5-03662be62eda')
embedding = CNCLIP_Embedding()

# # 上传和嵌入图片文件
# vector.upload_and_embed_files(embedding, 
#                               image_paths=image_paths, 
#                               save_path='/home/ubuntu/data/user01/codes/VideoRAG/image_data/1dcc108c-8bd4-42ad-b2c5-03662be62eda')

# 查询相似的图片
result, sim = vector.query(text_query='Did I leave the car door open?', EmbeddingModel=embedding, text_k=0, image_k=3)

print(result, '\n', sim)

# ['/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3179.png', '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3180.png', '/home/ubuntu/data/user01/codes/ego4d/seg_imgs_data_ffmpeg/1dcc108c-8bd4-42ad-b2c5-03662be62eda/frame3181.png'] 
#  [0.3936922300736449, 0.39045389177953294, 0.38463209139615795]
