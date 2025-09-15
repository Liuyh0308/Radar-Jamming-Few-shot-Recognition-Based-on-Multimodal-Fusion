import random
import numpy as np
import glob
from PIL import Image
import torch.nn.functional as F
import torch
import os
from scipy.io import loadmat

from core.dataset import MAMLDataset


class RJ_seq_Dataset(MAMLDataset):
    def get_file_list(self, data_path):
        """
        得到所有调制信号的类别文件夹名称
        传参在args.py中实现:
        数据集路径设置-----data_path: RML Data path
        返回一个含有所有调制信号类别名称的列表
        """
        return [f for f in
                glob.glob(data_path + "*", recursive=True)]  ## 对所有信噪比(trainingdta_alldb)进行训练 ，但测试是分别在不同信噪比上进行验证测试
        # return  [f for f in glob.glob(data_path + "*/", recursive=True)]    ## 划分信噪比训练（eg.训练集只有tariningdata_10db）

    def get_one_task_data(self):
        """
        本函数用于获取一个基本maml的task,maml以task为单位进行训练，
        这个task中包含了一个batchsize的support_set的序列与标签以及一个batchsize的query_set的序列与标签
        返回得到support_data, query_data

        """
        seq_dirs = random.sample(self.file_list,
                                 self.n_way)  ## 从读取到的file_list，filelist实际上是10种调制信号类别，在其中随机采样n_way个文件夹类别（n_way个类别的信号）
        support_data = []
        query_data = []

        support_sequence = []
        support_label = []
        query_sequence = []
        query_label = []

        for label, seq_dir in enumerate(seq_dirs):  ## 对每种信号进行
            seq_list = [f for f in glob.glob(seq_dir + "/*.mat", recursive=True)]  ## 对所有信噪比的数据进行训练

            sequences = random.sample(seq_list, self.k_shot + self.q_query)

            "1dcnn读取Seq"
            # 读取support set
            for seq_path in sequences[:self.k_shot]:
                mat_data = loadmat(seq_path)
                keys = [key for key in mat_data.keys() if not key.startswith("__")]  ##去除由‘loadmat’添加的特殊的键
                sequence = mat_data[keys[0]]  # 获取.mat第1个键所对应的数据，也就是‘J_fft’，就是序列数据
                sequence = np.array(sequence)  ## 得到（1,1024）的序列 但其中每个元素都为复数
                sequence = sequence.T    ## 得到（1024,1）的序列 但其中每个元素都为复数

                ## 对虚部和实部进行最大最小归一化 将元素值变换到0-1内，和图像元素值相同范围
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


                # # 提取幅值
                # magnitude = np.abs(sequence)
                # magnitude_min = np.min(magnitude)
                # magnitude_max = np.max(magnitude)
                # normalized_magnitude = (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
                #
                # # 提取相位
                # phase = np.angle(sequence)
                # phase_min = np.min(phase)
                # phase_max = np.max(phase)
                # normalized_phase = (phase - phase_min) / (phase_max - phase_min)
                #
                # # 合并幅值和相位，形成 (1024, 2) 的序列
                # sequence = np.concatenate((normalized_magnitude, normalized_phase), axis=-1)



                sequence = sequence.T  ## nn.conv1d的输入数据的维度应该是(batch_size, input_size, sequence_length) 调整数据维度顺序以是和卷积操作的形状，成为（bs,2,1024)
                support_data.append((sequence, label))

            # 读取query set
            for seq_path in sequences[self.k_shot:]:
                mat_data = loadmat(seq_path)
                keys = [key for key in mat_data.keys() if not key.startswith("__")]
                sequence = mat_data[keys[0]]
                sequence = np.array(sequence)  ## 得到（1,1024）的序列 但其中每个元素都为复数
                sequence = sequence.T

                ## 对虚部和实部进行最大最小归一化 将元素值变换到0-1内，和图像元素值相同范围
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
                                          axis=1)  ## 分离实部、虚部，得到（1024,2）的序列 sequence_length=1024 input_size=2


                # # 提取幅值
                # magnitude = np.abs(sequence)
                # magnitude_min = np.min(magnitude)
                # magnitude_max = np.max(magnitude)
                # normalized_magnitude = (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
                #
                # # 提取相位
                # phase = np.angle(sequence)
                # phase_min = np.min(phase)
                # phase_max = np.max(phase)
                # normalized_phase = (phase - phase_min) / (phase_max - phase_min)
                #
                # # 合并幅值和相位，形成 (1024, 2) 的序列
                # sequence = np.concatenate((normalized_magnitude, normalized_phase), axis=-1)



                sequence = sequence.T  ## nn.conv1d的输入数据的维度应该是(batch_size, input_size, sequence_length) 调整数据维度顺序以是和卷积操作的形状
                query_data.append((sequence, label))


        """
        最后得到的support_sequence为一个含有n_way个tensor的list列表
        eg. n_way=6时 label为0、 1、 2、 3、 4 、5  对random随机取出的n_way个调制信号种类进行数字编号

        同理可以知道query_set识别机制
        """

        # 打乱support set
        random.shuffle(support_data)
        for data in support_data:
            support_sequence.append(data[0])
            support_label.append(data[1])

        # 打乱query set
        random.shuffle(query_data)
        for data in query_data:
            query_sequence.append(data[0])
            query_label.append(data[1])

        return np.array(support_sequence), np.array(support_label), np.array(query_sequence), np.array(query_label)

