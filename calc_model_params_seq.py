import torch
from net.maml_1dcnn import ConvBlock, ConvBlockFunction,Classifier_1DCNN
from net.maml_MS1dcnn import ConvBlock, ConvBlock0, ConvBlockFunction, ConvBlockFunction0,Classifier_MS1DCNN


# 加载模型
# model_path = './model_path/best_seq_1dcnn.pth'
model_path = './model_path/best_seq_MS1dcnn.pth'

# 将模型加载到设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path).to(device)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')



# 假设输入张量的形状为(1, 2, 1024)
input_tensor = torch.randn(1, 2, 1024).to(device)

# 计算FLOPs
with torch.autograd.profiler.profile(enabled=True, use_cuda=torch.cuda.is_available()) as prof:
    model(input_tensor)

# 统计FLOPs
flops = sum([x.cpu_time_total for x in prof.function_events])  # 或者用其他相关方法获取FLOPs
print(f'Total FLOPs: {flops}')
