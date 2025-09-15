import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x


## 不用MAML方法 用普通深度学习的方法进行小样本任务训练 （用于对比用MAML与不用MAML方法的效果差别）
if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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
            image = image.resize((28, 28))
            image = np.array(image)
            image = np.expand_dims(image / 255., axis=0)
            image = torch.tensor(image, dtype=torch.float32)
            return image, label

        def load_images(self):
            images = []
            for class_name in self.classes:
                class_path = os.path.join(self.root_dir, class_name)
                class_images = []
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    class_images.append((img_path, self.class_to_idx[class_name]))
                images.extend(class_images)
            return images




    # 数据集根目录
    train_root_dir = './dataset/Radar_Jamming_Signal_Dataset/Trainning_data/dataset_img/All_dB'
    val_root_dir = './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/10_dB'
    # val_root_dir = './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/0_dB'

    # 创建数据集实例
    train_dataset = CustomDataset(root_dir=train_root_dir)
    val_dataset = CustomDataset(root_dir=val_root_dir)


    train_batch_size = 128
    val_batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    # 定义简单的4层CNN模型


    # 实例化模型
    model = SimpleCNN()
    if torch.cuda.is_available():
        model = model.to('cuda')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    # 训练模型
    num_epochs = 100
    total_acc = []

    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        total_acc.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {loss.item():.4f}, Validation Loss: {average_val_loss:.4f}, Val_Accuracy: {accuracy * 100:.2f}%')

    print(total_acc)

    torch.save(model, './model_path/best_base_without_maml.pth')

