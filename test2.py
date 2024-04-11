from RAG.Embeddings import CNCLIP_Embedding, BaseEmbeddings
from PIL import Image
import requests

embedding = CNCLIP_Embedding()

url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_feature = embedding.get_embedding(image=image)
texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
# text_feature = embedding.get_embedding(text=texts[0])
text_features = []
for text in texts:
    text_features.append(embedding.get_embedding(text=text))
print(image_feature.__len__())
# print(text_feature.__len__())
for text_feature in text_features:
    print(BaseEmbeddings.cosine_similarity(image_feature, text_feature))
