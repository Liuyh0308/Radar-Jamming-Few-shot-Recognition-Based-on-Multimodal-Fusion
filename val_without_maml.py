import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
from train_without_maml_base import SimpleCNN
from train_without_maml_resnet import resnet, ConvBlock, ResidualBlock, ResidualBlock1


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
        image = Image.open(img_path)
        # image = image.resize((28, 28))      ## resnet需注释
        image = np.array(image)
        image = np.expand_dims(image / 255., axis=0)  # Normalize and add channel dimension
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

# Load the model
# model = torch.load('./model_path/best_base_without_maml.pth')
model = torch.load('./model_path/best_resnet_without_maml.pth')


model.eval()

if torch.cuda.is_available():
    model.cuda()

# Validation datasets and loaders for different SNR levels
snr_levels = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
val_data_dirs = {
    snr: f'./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/{snr}_dB/' for snr in snr_levels
    # snr: f'./dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/{snr}_dB/' for snr in snr_levels
}

valid_loaders = {
    snr: DataLoader(CustomDataset(root_dir=val_data_dirs[snr]), batch_size=128, shuffle=False)
    for snr in snr_levels
}

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion.cuda()

# Evaluate the model on each SNR level
results = {}
for snr, loader in valid_loaders.items():
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_val_loss = val_loss / len(loader)
    results[snr] = {'Validation Loss': average_val_loss, 'Accuracy': accuracy * 100}

# Print results for each SNR level
for snr, metrics in results.items():
    print(f'SNR {snr}dB: Loss = {metrics["Validation Loss"]:.4f}, Accuracy = {metrics["Accuracy"]:.2f}%')
