import random
import numpy as np
import glob
from PIL import Image
import torch.nn.functional as F
import torch
import os
import h5py
from scipy.io import loadmat

from core.dataset import MAMLDataset0


class RJ_fusion_Dataset(MAMLDataset0):
    def get_file_list(self, data_path):
        """
        得到所有调制信号的类别文件夹名称
        传参在args.py中实现:
        数据集路径设置-----data_path: RML Data path
        返回一个含有所有调制信号类别名称的列表
        """
        return [f for f in glob.glob(data_path + "*", recursive=True)]  ## 对所有信噪比(trainingdta_alldb)进行训练 ，但测试是分别在不同信噪比上进行验证测试


    def get_one_task_data(self):
        """
        本函数用于获取一个基本maml的task,maml以task为单位进行训练，
        这个task中包含了一个batchsize的support_set的图片与标签以及一个batchsize的query_set的图片与标签
        返回得到support_data, query_data
        """

        img_dirs = self.file_list1    ## 从读取到的file_list，filelist实际上是10种调制信号类别，在其中随机采样n_way个文件夹类别（n_way个类别的信号）
        seq_dirs = self.file_list2

        """
        随机抽取N-way信号种类
        """
        # 根据图片文件目录随机抽样
        sampled_img_dirs = random.sample(img_dirs, self.n_way)

        # 使用相同的索引或名称从序列文件目录中抽样
        sampled_seq_dirs = [seq_dirs[img_dirs.index(dir_)] for dir_ in sampled_img_dirs]


        """
        固定选取N-way个信号种类,用于val时观察各个信号类别在不同SNR下的acc
        """
        # # 想固定选择的n_way个类别目录名

        # fixed_class_names = ['AM', 'FM', 'MNJ', 'SMSP', 'VMT']
        # fixed_class_names = ['ISRJ', 'R_VGPO', 'RGPO', 'RMT', 'VMT']
        # fixed_class_names = ['COMB', 'ISRJ', 'MNJ', 'VMT', 'RGPO']


        # fixed_img_dirs = [dir_ for dir_ in img_dirs
        #                                 if any(class_name in os.path.basename(dir_) for class_name in fixed_class_names)]
        # # # 使用相同的索引或名称从序列文件目录中抽样
        # fixed_seq_dirs = [seq_dirs[img_dirs.index(dir_)] for dir_ in fixed_img_dirs]



        support_data_img = []
        query_data_img = []
        support_data_seq = []
        query_data_seq = []


        support_image = []
        support_img_label = []
        query_image = []
        query_img_label = []

        support_sequence = []
        support_seq_label = []
        query_sequence = []
        query_seq_label = []


## 读取图片类&序列类数据集
        ##随机
        for label, (img_dir, seq_dir) in enumerate(zip(sampled_img_dirs, sampled_seq_dirs)):

        ##固定
        # for label, (img_dir, seq_dir) in enumerate(zip(fixed_img_dirs, fixed_seq_dirs)):

            # 处理图片数据
            img_list = [f for f in glob.glob(img_dir + "/*.png", recursive=True)]  ## 对所有信噪比的数据进行训练
            images = random.sample(img_list, self.k_shot + self.q_query)
            # 读取image support set
            for img_path in images[:self.k_shot]:
                image = Image.open(img_path).convert('L')
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                support_data_img.append((image, label))
            # 读取image query set
            for img_path in images[self.k_shot:]:
                image = Image.open(img_path).convert('L')
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                query_data_img.append((image, label))

            # 处理序列数据
            seq_list = [f for f in glob.glob(seq_dir + "/*.mat", recursive=True)]
            # 获取被选中的图像文件的索引
            selected_indices = [img_list.index(img_path) for img_path in images]
            # 使用这些索引来选择相应的序列文件
            sequences = [seq_list[i] for i in selected_indices]

            # # 读取sequence support set
            for seq_path in sequences[:self.k_shot]:
                mat_data = loadmat(seq_path)
                keys = [key for key in mat_data.keys() if not key.startswith("__")]  ##去除由‘loadmat’添加的特殊的键
                sequence = mat_data[keys[0]]    # 获取.mat第二个键所对应的数据，也就是‘wav’，就是序列数据
                sequence = np.array(sequence)      ## 得到（1024,1）的序列 但其中每个元素都为复数
                sequence = sequence.T

            ## 对虚部和实部进行最大最小归一化 将元素值变换到0-1内，和图像元素值相同范围
                real_part = np.real(sequence)     ## 提取序列中每个元素的实部
                real_part_min = np.min(real_part)
                real_part_max = np.max(real_part)
                normalized_real_part = (real_part-real_part_min)/(real_part_max-real_part_min)

                imaginary_part = np.imag(sequence)     ## 提取序列中每个元素的虚部
                imaginary_part_min = np.min(imaginary_part)
                imaginary_part_max = np.max(imaginary_part)
                normalized_imaginary_part = (imaginary_part - imaginary_part_min) / (imaginary_part_max - imaginary_part_min)

                # real_part_1 = np.sqrt(normalized_real_part ** 2 + normalized_imaginary_part ** 2)
                # imaginary_part_1 = np.arctan(normalized_imaginary_part/ normalized_real_part)

                sequence = np.concatenate((normalized_real_part, normalized_imaginary_part), axis=1)    ## 分离实部、虚部，得到（1024,2）的序列 sequence_length=1024 input_size=2
                sequence = sequence.T              ## nn.conv1d的输入数据的维度应该是(batch_size, input_size, sequence_length) 调整数据维度顺序以是和卷积操作的形状
                support_data_seq.append((sequence, label))

            # # 读取sequence query set
            for seq_path in sequences[self.k_shot:]:
                mat_data = loadmat(seq_path)
                keys = [key for key in mat_data.keys() if not key.startswith("__")]
                sequence = mat_data[keys[0]]
                sequence = np.array(sequence)    ## 得到（1024,1）的序列 但其中每个元素都为复数
                sequence = sequence.T

            ## 对虚部和实部进行最大最小归一化 将元素值变换到0-1内，和图像元素值相同范围
                real_part = np.real(sequence)  ## 提取序列中每个元素的实部
                real_part_min = np.min(real_part)
                real_part_max = np.max(real_part)
                normalized_real_part = (real_part - real_part_min) / (real_part_max - real_part_min)

                imaginary_part = np.imag(sequence)     ## 提取序列中每个元素的虚部
                imaginary_part_min = np.min(imaginary_part)
                imaginary_part_max = np.max(imaginary_part)
                normalized_imaginary_part = (imaginary_part - imaginary_part_min) / (imaginary_part_max - imaginary_part_min)

                # real_part_1 = np.sqrt(normalized_real_part ** 2 + normalized_imaginary_part ** 2)
                # imaginary_part_1 = np.arctan(normalized_imaginary_part/ normalized_real_part)

                sequence = np.concatenate((normalized_real_part, normalized_imaginary_part), axis=1)   ## 分离实部、虚部，得到（1024,2）的序列 sequence_length=1024 input_size=2
                sequence = sequence.T      ## nn.conv1d的输入数据的维度应该是(batch_size, input_size, sequence_length) 调整数据维度顺序以是和卷积操作的形状
                query_data_seq.append((sequence, label))


        ## 创建一个索引列表，与support和query数据对应
        indexes = list(range(len(support_data_img)))
        ## 打乱索引
        random.shuffle(indexes)

        ## 使用打乱的索引对image与sequence的support数据进行排序
        support_data_img = [support_data_img[i] for i in indexes]
        support_data_seq = [support_data_seq[i] for i in indexes]

        ## 使用打乱的索引对image与sequence的query数据进行排序
        indexes = list(range(len(query_data_img)))
        random.shuffle(indexes)
        query_data_img = [query_data_img[i] for i in indexes]
        query_data_seq = [query_data_seq[i] for i in indexes]

        ## 提取图片数据集的图片和标签
        for data in support_data_img:
            support_image.append(data[0])
            support_img_label.append(data[1])
        for data in query_data_img:
            query_image.append(data[0])
            query_img_label.append(data[1])

        ## 提取序列数据集的序列和标签
        for data in support_data_seq:
            support_sequence.append(data[0])
            support_seq_label.append(data[1])

        for data in query_data_seq:
            query_sequence.append(data[0])
            query_seq_label.append(data[1])


        return np.array(support_image), np.array(support_img_label), np.array(query_image), np.array(query_img_label), \
               np.array(support_sequence), np.array(support_seq_label), np.array(query_sequence), np.array(query_seq_label)



