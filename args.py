import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from core.helper import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')



"""
雷达干扰数据集的training_alldb中的每种调制信号下的顺序分布排列为-20db,-18db.....-10db,然后是10db,-8db,8db,-6db,6db....（依次交叉）..-2db,2db,0db;
不论是dataset_image or dataset_sequence都是如此排列training_alldb的数据
"""

## 单独的RJ_fusion图片数据集
parser.add_argument('--train_data_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Trainning_data/dataset_img/All_dB/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/10_dB/",
                    help='The directory containing the validation image data.')



## 单独的RJ_fusion序列数据集
# parser.add_argument('--train_data_dir', type=str,
#                     default="./dataset/Radar_Jamming_Signal_Dataset/Trainning_data/dataset_seq/All_dB/",
#                     help='The directory containing the train sequence data.')
# parser.add_argument('--val_data_dir', type=str,
#                     default="./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/10_dB/",
#                     help='The directory containing the validation sequence data.')


## 多模态RJ_fusion数据集传参
## 图片数据集传参
parser.add_argument('--train_data_img_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Trainning_data/dataset_img/All_dB/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_img_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/10_dB/",
                    help='The directory containing the validation image data.')
# 序列数据集传参
parser.add_argument('--train_data_seq_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Trainning_data/dataset_seq/All_dB/",
                    help='The directory containing the train sequence data.')
parser.add_argument('--val_data_seq_dir', type=str,
                    default="./dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/10_dB/",
                    help='The directory containing the validation sequence data.')



## 图域时频图单模态超参数
# parser.add_argument('--task_num', type=int, default=8,          ## batchsize
#                     help='Number of task per train batch.')
# parser.add_argument('--val_task_num', type=int, default=4,      ## batchsize
#                     help='Number of task per test batch.')
# parser.add_argument('--num_workers', type=int, default=4, help='The number of torch dataloader thread.')
#
# parser.add_argument('--epochs', type=int, default=200,
#                     help='The training epochs.')
#
# # parser.add_argument('--inner_lr', type=float, default=0.02,
# #                     help='The learning rate of the support set.')
# parser.add_argument('--inner_lr', type=float, default=0.01,
#                     help='The learning rate of the support set.')
# parser.add_argument('--outer_lr', type=float, default=0.005,
#                     help='The learning rate of the query set.')


## I/Q波形域序列单模态超参数
# parser.add_argument('--task_num', type=int, default=10,          ## batchsize
#                     help='Number of task per train batch.')
# parser.add_argument('--val_task_num', type=int, default=4,      ## batchsize
#                     help='Number of task per test batch.')
# parser.add_argument('--num_workers', type=int, default=4, help='The number of torch dataloader thread.')
#
# parser.add_argument('--epochs', type=int, default=2000,
#                     help='The training epochs.')
# parser.add_argument('--inner_lr', type=float, default=0.02,
#                     help='The learning rate of the support set.')
# parser.add_argument('--outer_lr', type=float, default=0.005,
#                     help='The learning rate of the query set.')


# # 多模态超参数
parser.add_argument('--task_num', type=int, default=8,          ## batchsize
                    help='Number of task per train batch.')
parser.add_argument('--val_task_num', type=int, default=4,      ## batchsize
                    help='Number of task per test batch.')
parser.add_argument('--num_workers', type=int, default=4, help='The number of torch dataloader thread.')

parser.add_argument('--epochs', type=int, default=100,
                    help='The training epochs.')
parser.add_argument('--inner_lr', type=float, default=0.02,
                    help='The learning rate of the support set.')
parser.add_argument('--outer_lr', type=float, default=0.01,
                    help='The learning rate of the query set.')


##  传参  单个任务分类数N/单个任务支持集图片数K/单个任务查询集集图片数Q
parser.add_argument('--n_way', type=int, default=5,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=3,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=4,
                    help='The number of query set image for every task.')



parser.add_argument('--summary_path', type=str,
                    default="./summary",
                    help='The directory of the summary writer.')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(1206)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
