import torch

# 输入张量
input_tensor = torch.tensor([[29.3215, 30.4848, 28.5620, 34.3111]])

# 计算softmax
softmax_output = torch.softmax(input_tensor, dim=1)

# 计算每个元素的指数
exp_tensor = torch.exp(input_tensor)

# 计算指数和
sum_exp = torch.sum(exp_tensor, dim=1)

# 计算softmax的每个元素
calculated_softmax = exp_tensor / sum_exp.unsqueeze(1)

print(softmax_output, calculated_softmax)
