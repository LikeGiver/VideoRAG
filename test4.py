from RAG.VectorBase import VectorStore
# from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import CNCLIP_Embedding

vector = VectorStore()

embedding = CNCLIP_Embedding()

vector.upload_and_embed_files(embedding, document=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], 
                              image_paths=['/home/ubuntu/data/user01/codes/VideoRAG/image_data/pokemon.jpeg', 
                                           '/home/ubuntu/data/user01/codes/VideoRAG/image_data/d0ee81f16175c97770192fb691fdda8da1f4f349.png'])

result = vector.query(text_query="pokemon", EmbeddingModel=embedding, k=5)

print(result)