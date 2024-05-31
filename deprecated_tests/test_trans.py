import json
import time
from googletrans import Translator
from tqdm import tqdm

# 初始化翻译器
translator = Translator()

# 读取JSON文件
input_file = '/home/ubuntu/data/user01/codes/ego4d/data/unified/annotations.QaEgo4D_test.json'
output_file = 'translated_data.json'

with open(input_file, 'r') as file:
    data = json.load(file)

# 读取已翻译的数据（如果存在）
try:
    with open(output_file, 'r', encoding='utf-8') as file:
        translated_data = json.load(file)
except FileNotFoundError:
    translated_data = []

# 获取已经翻译过的项数
translated_count = len(translated_data)

# 翻译函数，带重试机制
def translate_text(text, dest_language='zh-cn', retries=3):
    for _ in range(retries):
        try:
            translated = translator.translate(text, dest=dest_language)
            return translated.text
        except Exception as e:
            print(f"Error translating text: {e}")
            time.sleep(5)
    return text  # 返回原文本以继续处理

# 继续翻译未完成的部分
for item in tqdm(data[translated_count:], desc="Translating", unit="item"):
    if 'answer' in item:
        item['answer'] = translate_text(item['answer'])
    if 'question' in item:
        item['question'] = translate_text(item['question'])
    translated_data.append(item)

    # 每翻译完成一个item就保存一次
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

print("翻译完成，结果已保存到translated_data.json")