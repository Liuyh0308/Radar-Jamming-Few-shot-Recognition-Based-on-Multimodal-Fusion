import random
import numpy as np
import glob
from PIL import Image
import torch.nn.functional as F
import torch
import os

from core.dataset import MAMLDataset


class RJ_img_Dataset(MAMLDataset):
    def get_file_list(self, data_path):
        """
        得到所有调制信号的类别文件夹名称
        传参在args.py中实现:
        数据集路径设置-----data_path: RML Data path
        返回一个含有所有调制信号类别名称的列表
        """
        return [f for f in glob.glob(data_path + "*", recursive=True)]        ## 对所有信噪比(trainingdta_alldb)进行训练 ，但测试是分别在不同信噪比上进行验证测试
        # return  [f for f in glob.glob(data_path + "*/", recursive=True)]    ## 划分信噪比训练（eg.训练集只有tariningdata_10db）


    def get_one_task_data(self):
        """
        本函数用于获取一个基本maml的task,maml以task为单位进行训练，
        这个task中包含了一个batchsize的support_set的图片与标签以及一个batchsize的query_set的图片与标签
        返回得到support_data, query_data

        """
        img_dirs = random.sample(self.file_list, self.n_way)    ## 从读取到的file_list，filelist实际上是10种调制信号类别，在其中随机采样n_way个文件夹类别（n_way个类别的信号）
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs):         ## 对每种信号进行
            img_list = [f for f in glob.glob(img_dir + "/*.png", recursive=True)]     ## 对所有信噪比的数据进行训练


            images = random.sample(img_list, self.k_shot + self.q_query)

            # 读取support set
            for img_path in images[:self.k_shot]:
                image = Image.open(img_path).convert('L')     #可转灰度图  单通道

                ## 对图片大小进行调整（适应不同网络）
                # image = image.resize((112, 112))
                # image = image.resize((28, 28))      ## 用四层简单cnn的情况下 imagesize为（28,28）  baseline
                # image = image.resize((56, 56))     ## senet
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                support_data.append((image, label))

            # 读取query set
            for img_path in images[self.k_shot:]:
                image = Image.open(img_path).convert('L')    #可转灰度图，由此图片成为单通道224*224*1

                ## 对图片大小进行调整（适应不同网络）
                # image = image.resize((112, 112))
                # image = image.resize((28, 28))    ## 四层简单cnn  baseline
                # image = image.resize((56, 56))     ## SEnet
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                query_data.append((image, label))


        """
        最后得到的support_image为一个含有n_way个tensor的list列表
        eg. n_way=6时 label为0、 1、 2、 3、 4 、5  对random随机取出的n_way个调制信号种类进行数字编号
        
        同理可以知道query_set识别机制
        """

        # 打乱support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # 打乱query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

