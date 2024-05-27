# VideoRAG
本项目探索针对长视频对话的RAG系统，旨在实现第一人称视角下的长视频记忆系统。

## TODO
1. 扩展数据库到支持图片（存储以及混合模态查询，基于Chinese-CLIP）✅
2. 测试基于视频抽帧的RAG效果(基于Qwen-VL)

## refer projects
1. TinyRAG
2. LlamaIndex
3. Chinese CLIP
4. Qwen-VL (huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen-VL --local-dir ./Qwen-VL)

## env preparation

### Qwen-VL:
pip install transformers_stream_generator
pip install accelerate