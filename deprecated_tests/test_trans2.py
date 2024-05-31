import json
from googletrans import Translator
from tqdm import tqdm

# 初始化翻译器
translator = Translator()

# 翻译函数
def translate_text(text, dest_language='zh-cn'):
    translated = translator.translate(text, dest=dest_language)
    return translated.text

print(translate_text("hello, I am yike!"))