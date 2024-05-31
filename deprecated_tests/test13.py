import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

# 分词处理
def tokenize(text):
    return ' '.join(jieba.cut(text))

# 计算相似度分数
def calculate_scores(answer, response):
    answer_tokens = tokenize(answer)
    response_tokens = tokenize(response)
    
    # BLEU 分数
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([answer_tokens.split()], response_tokens.split(), smoothing_function=smoothie)
    
    # METEOR 分数
    meteor = single_meteor_score(set(answer), set(response))
    
    # ROUGE-L 分数
    rouge = Rouge()
    scores = rouge.get_scores([response_tokens], [answer_tokens])
    rougeL_fmeasure = scores[0]['rouge-l']['f']
    
    return bleu_score, meteor, rougeL_fmeasure

# 示例
question = "她放下帽子后，帽子在哪里？"
answer = "在衣柜里"
response = "在柜子里。"

bleu, meteor, rougeL = calculate_scores(answer, response)
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Response: {response}")
print(f"BLEU: {bleu}")
print(f"METEOR: {meteor}")
print(f"ROUGE-L: {rougeL}")

# 检查平均分
scores = {'bleu': [], 'meteor': [], 'rougeL_fmeasure': []}
scores['bleu'].append(bleu)
scores['meteor'].append(meteor)
scores['rougeL_fmeasure'].append(rougeL)

avg_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}
print("Average Scores:", avg_scores)