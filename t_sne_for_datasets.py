import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# def load_data(root_dir):
#     data = []
#     labels = []
#
#     # 遍历根目录下的所有信噪比文件夹
#     snr_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#     for snr in snr_dirs:
#         snr_path = os.path.join(root_dir, snr)
#         # 假设每个信噪比文件夹下都有若干表示不同信号类型的子文件夹
#         signal_type_dirs = [d for d in os.listdir(snr_path) if os.path.isdir(os.path.join(snr_path, d))]
#         for signal_type in signal_type_dirs:
#             signal_folder = os.path.join(snr_path, signal_type)
#             for file_name in os.listdir(signal_folder):
#                 file_path = os.path.join(signal_folder, file_name)
#                 try:
#                     image = Image.open(file_path).convert('L')  # 转换为灰度图
#                     image = image.resize((28, 28))  # 调整尺寸，例如 28x28
#                     image_array = np.array(image).flatten()  # 将图像拉平为一个向量
#                     data.append(image_array)
#                     # 这里可以根据需要选择标签：仅用信号类型，或 SNR 与信号类型组合
#                     # 示例中使用 SNR 与信号类型的组合：
#                     labels.append(signal_type)
#
#
#                 except Exception as e:
#                     print(f"处理文件 {file_path} 时发生错误: {e}")
#                     continue
#     return np.array(data), np.array(labels)


def load_data(root_dir):
    data = []
    labels = []

    # 遍历根目录下的所有信噪比文件夹
    snr_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for snr in snr_dirs:
        snr_path = os.path.join(root_dir, snr)
        # 假设每个信噪比文件夹下都有若干表示不同信号类型的子文件夹
        signal_type_dirs = [d for d in os.listdir(snr_path) if os.path.isdir(os.path.join(snr_path, d))]
        for signal_type in signal_type_dirs:
            signal_folder = os.path.join(snr_path, signal_type)
            for file_name in os.listdir(signal_folder):
                file_path = os.path.join(signal_folder, file_name)
                try:
                    mat_data = loadmat(file_path)
                    keys = [key for key in mat_data.keys() if not key.startswith("__")]  ##去除由‘loadmat’添加的特殊的键
                    sequence = mat_data[keys[0]]  # 获取.mat第1个键所对应的数据，也就是‘J_fft’，就是序列数据
                    sequence = np.array(sequence)  ## 得到（1,1024）的序列 但其中每个元素都为复数
                    sequence = sequence.T  ## 得到（1024,1）的序列 但其中每个元素都为复数

                    real_part = np.real(sequence)  ## 提取序列中每个元素的实部
                    real_part_min = np.min(real_part)
                    real_part_max = np.max(real_part)
                    normalized_real_part = (real_part - real_part_min) / (real_part_max - real_part_min)

                    imaginary_part = np.imag(sequence)  ## 提取序列中每个元素的虚部
                    imaginary_part_min = np.min(imaginary_part)
                    imaginary_part_max = np.max(imaginary_part)
                    normalized_imaginary_part = (imaginary_part - imaginary_part_min) / (
                            imaginary_part_max - imaginary_part_min)

                    sequence = np.concatenate((normalized_real_part, normalized_imaginary_part),
                                              axis=1)  ## 将实部、虚部拼接得到（bs,1024,2）的序列 sequence_length=1024 input_size=2

                    sequence = sequence.T  ## nn.conv1d的输入数据的维度应该是(batch_size, input_size, sequence_length) 调整数据维度顺序以是和卷积操作的形状，成为（bs,2,1024)

                    sequence_array = np.array(sequence).flatten()  # 将图像拉平为一个向量

                    data.append(sequence_array)

                    labels.append(signal_type)


                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误: {e}")
                    continue
    return np.array(data), np.array(labels)




# 修改数据根路径为 dataset_img 文件夹
# data_dir = './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img'
data_dir = './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq'

data, labels = load_data(data_dir)

# 进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data)

# 可视化结果
plt.figure(figsize=(10, 8))
unique_labels = list(set(labels))
colors = plt.cm.get_cmap('tab10', len(unique_labels))

for i, label in enumerate(unique_labels):
    indices = labels == label
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                color=colors(i), label=label)

# plt.legend()
# plt.title('所有SNR下的干扰信号样本t-SNE可视化', fontsize=12)
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
plt.show()
