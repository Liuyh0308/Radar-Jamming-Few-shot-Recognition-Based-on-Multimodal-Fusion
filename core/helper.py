import shutil

from net.maml import Classifier
from net.maml_senet import SE_Classifier
from net.maml_resnet import Res_Classifier
from net.maml_concat import Concat_Classifier
from net.maml_se_resnet import SE_Res_Classifier
from net.maml_1dcnn import Classifier_1DCNN
from net.maml_MS1dcnn import Classifier_MS1DCNN
from net.maml_fusion_net import Fusion_Classifier
from net.maml_lstm_res18_fusionnet import LSTM_Res18_Fusion_Classifier
from net.maml_DCFAnet import DCFA_Classifier
from net.maml_DCFAproF import DCFAproF_Classifier

from core.dataset import OmniglotDataset, RJ_img_Dataset, RJ_seq_Dataset, RJ_fusion_Dataset


import os
import torch
from torch import nn
import numpy as np
import random


def get_model(args, dev):
    """
    获取所需要使用到的模型backbone
    传参设置:
    args: ArgumentParser
    dev: torch dev
    返回得到模型model
    """
    ## 单模态模型
    ## 图域      输入通道为1 图片为灰度图
    # model = Classifier(1, args.n_way).cuda()
    # model = SE_Classifier(1, args.n_way).cuda()
    # model = SE_Res_Classifier(1, args.n_way).cuda()
    # model = Res_Classifier(1,  args.n_way).cuda()
    # model = Concat_Classifier(1, args.n_way).cuda()

    ## I/Q域   输入通道为2  I/Q两个通道
    # model = Classifier_1DCNN(2, args.n_way).cuda()
    # model = Classifier_MS1DCNN(2, args.n_way).cuda()

    ## 多模态模型
    # model = Fusion_Classifier(1, 2, args.n_way).cuda()
    # model = LSTM_Res18_Fusion_Classifier(1, 2, args.n_way).cuda()
    # model = DCFA_Classifier(1, 2, args.n_way).cuda()
    model = DCFAproF_Classifier(1, 2, args.n_way).cuda()


    model.to(dev)

    return model



# 得到MAML Omniglot 数据集
# def get_dataset(args):
#     """
#     得到Omniglot数据集
#     传参设置:
#     args: ArgumentParser（参数解释器）
#     返回: dataset数据集
#     """
#     train_dataset = OmniglotDataset(args.train_data_dir, args.task_num,
#                                     n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
#     val_dataset = OmniglotDataset(args.val_data_dir, args.val_task_num,
#                                   n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
#
#     return train_dataset, val_dataset



# 得到RJ_img数据集
def get_dataset(args):
    """
    得到RJ image 数据集
    传参设置:
    args: ArgumentParser（参数解释器）
    返回: dataset数据集
    """

    train_dataset = RJ_img_Dataset(args.train_data_dir, args.task_num,
                                    n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
    val_dataset = RJ_img_Dataset(args.val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return train_dataset, val_dataset


## 得到RJ_seq数据集
# def get_dataset(args):
#     """
#     得到RJ sequence 数据集
#     传参设置:
#     args: ArgumentParser（参数解释器）
#     返回: dataset数据集
#     """
#
#     train_dataset = RJ_seq_Dataset(args.train_data_dir, args.task_num,
#                                     n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
#     val_dataset = RJ_seq_Dataset(args.val_data_dir, args.val_task_num,
#                                   n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
#
#     return train_dataset, val_dataset


# 得到RJ_fusion数据集
def get_dataset_fusion(args):
    """
    得到RJ数据集
    传参设置:
    args: ArgumentParser（参数解释器）
    返回: dataset数据集
    """

    train_dataset = RJ_fusion_Dataset(args.train_data_img_dir, args.train_data_seq_dir, args.task_num,
                                    n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
    val_dataset = RJ_fusion_Dataset(args.val_data_img_dir, args.val_data_seq_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return train_dataset, val_dataset


### 用于将预训练模型在各个信噪比上进行测试
def get_val_img_dataset(snr, args):
    """
    返回: val_img_dataset数据集进行测试test用
    """
    val_data_dir = args.val_data_dirs[snr]     # 假设 args.val_data_dirs 是一个字典，映射信噪比到数据路径
    val_dataset = RJ_img_Dataset(val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return val_dataset

def get_val_seq_dataset(snr, args):
    """
    返回: val_seq_dataset数据集进行测试test用
    """
    val_data_dir = args.val_data_dirs[snr]  # 假设 args.val_data_dirs 是一个字典，映射信噪比到数据路径
    val_dataset = RJ_seq_Dataset(val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return val_dataset


def get_val_fusion_dataset(snr, args):
    """
    返回: val_seq_dataset数据集进行测试test用
    """
    val_data_img_dir = args.val_data_img_dirs[snr]  # 假设 args.val_data_dirs 是一个字典，映射信噪比到数据路径
    val_data_seq_dir = args.val_data_seq_dirs[snr]
    val_dataset = RJ_fusion_Dataset(val_data_img_dir, val_data_seq_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return val_dataset


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")


def adjust_learning_rate1(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every ** epochs"""
    lr = start_lr * (0.1 ** (epoch // 50))
    # lr = start_lr * (0.1 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every ** epochs"""
    lr = start_lr * (0.1 ** (epoch // 50))
    # lr = start_lr * (0.1 ** (epoch // 70))
    return lr


## 此函数将输入混淆矩阵，并输出计算的准确率、精确率、召回率和F1分数
def compute_metrics(confusion_matrix):
    with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除以零的警告
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0).astype(np.float64)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1).astype(np.float64)
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix).astype(np.float64)
        f1_scores = 2 * (precision * recall) / (precision + recall)

        precision = np.nan_to_num(precision)  # 将NaN转为0
        recall = np.nan_to_num(recall)
        f1_scores = np.nan_to_num(f1_scores)

    return accuracy, precision, recall, f1_scores