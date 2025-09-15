import matplotlib
import numpy as np
import torch
import random
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号


# 定义 CustomDataset 类
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')
        # image = image.resize((28, 28))  ## 如果使用ResNet请注释掉这一行
        image = np.array(image)
        image = np.expand_dims(image / 255., axis=0)  # 归一化并添加通道维度
        image = torch.tensor(image, dtype=torch.float32)
        return image, label

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                images.append((img_path, self.class_to_idx[class_name]))
        return images

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualBlock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock1, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.conv1 = ConvBlock(1, 64)
        self.conv2 = ResidualBlock(64, 64)
        self.conv3 = ResidualBlock(64, 64)
        self.conv4 = ResidualBlock1(64, 128)
        self.conv5 = ResidualBlock(128, 128)
        self.conv6 = ResidualBlock1(128, 256)
        self.conv7 = ResidualBlock(256, 256)
        self.conv8 = ResidualBlock1(256, 512)
        self.conv9 = ResidualBlock(512, 512)
        self.pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512, 13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x1 = self.pool(x)
        x1 = x1.view(x1.shape[0], -1)
        features = x1
        x1 = self.fc1(x1)
        return x1, features

# 加载模型
model = torch.load('./model_path/best_resnet_without_maml.pth')
model.eval()

if torch.cuda.is_available():
    model.cuda()

# 为不同SNR级别创建验证数据集和数据加载器
snr_levels = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
# snr_levels = [-2, 0, 2, 4, 6, 8, 10]
# snr_levels = [-20, -18, -16, -14, -12, -10]

val_data_dirs = {
    snr: f'./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/{snr}_dB/' for snr in snr_levels
}

valid_loaders = {
    snr: DataLoader(CustomDataset(root_dir=val_data_dirs[snr]), batch_size=128, shuffle=False)
    for snr in snr_levels
}

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion.cuda()

# 评估每个SNR级别的模型
results = {}
feature_vectors = []
labels_list = []

for snr, loader in valid_loaders.items():
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs, features = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            feature_vectors.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    accuracy = correct / total
    average_val_loss = val_loss / len(loader)
    results[snr] = {'Validation Loss': average_val_loss, 'Accuracy': accuracy * 100}

# 拼接特征向量和标签
feature_vectors = np.concatenate(feature_vectors)
labels_list = np.concatenate(labels_list)



# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(feature_vectors)

# 可视化
plt.figure(figsize=(10, 8))
for label in np.unique(labels_list):
    indices = np.where(labels_list == label)
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=str(label), alpha=0.5)
# plt.legend()
plt.title('高维向量的t-SNE可视化' ,fontsize=16)
plt.show()

# 打印每个SNR级别的结果
for snr, metrics in results.items():
    print(f'SNR {snr}dB: Loss = {metrics["Validation Loss"]:.4f}, Accuracy = {metrics["Accuracy"]:.2f}%')
