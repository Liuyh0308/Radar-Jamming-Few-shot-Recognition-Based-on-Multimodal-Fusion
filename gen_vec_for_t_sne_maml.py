import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import RML_img_Dataset

from core.helper import get_val_fusion_dataset, compute_metrics

##  选择模型
from net.maml_fusion_net import maml_fusion_train
from net.maml_lstm_res18_fusionnet import maml_lstm_res18_fusion_train
from net.maml_DCFAnet import maml_DCFA_train
from net.maml_DCFAproF import maml_DCFAproF_train, maml_DCFAproF_train1

from args import args, dev

if __name__ == '__main__':

    model = torch.load('./model_path/best_fusion_DCFAproF.pth')  ## 效果较好
    # model = torch.load('./model_path/best_fusion_DCFAproF1.pth')

    model.eval()  # 将模型设置为评估模式

    params = [p for p in model.parameters() if p.requires_grad]
    start_lr1 = args.outer_lr
    start_lr2 = args.inner_lr
    optimizer = optim.Adam(params, start_lr1)

    # 创建一个信噪比到数据路径的映射
    # 验证集z
    args.val_data_img_dirs = {
       10 : './data/Modulation_fusion_train/dataset_images/trainingdata_alldb/'
    }

    args.val_data_seq_dirs = {
       10:  './data/Modulation_fusion_train/dataset_sequences/trainingdata_alldb/'

    }

    # 定义一个空列表，用于存储所有SNR下的特征
    all_features = []
    all_labels = []  # 用于存储所有SNR下的标签


    valid_loaders = []

    snr = 10


    dataset = get_val_fusion_dataset(snr, args)
    loader = DataLoader(dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)


    # 初始化存储当前SNR下特征的列表
    features_snr = []
    labels_snr = []  # 存储当前SNR下的标签


    for support_images, support_img_labels, query_images, query_img_labels, \
            support_sequences, support_seq_labels, query_sequences, query_seq_labels in loader:

            # 得到变量
            support_images = support_images.float().to(dev)
            # support_img_labels = support_img_labels.long().to(dev)
            support_labels = support_img_labels.long().to(
                dev)  ## 对图片与序列support集与query集都只返回一个总label，而不去区分support_img_label or query_img_label之类

            query_images = query_images.float().to(dev)
            # query_img_labels = query_img_labels.long().to(dev)
            query_labels = query_img_labels.long().to(dev)  ## 对图片与序列support集与query集都只返回一个总label，而不去区分support_img_label or query_img_label之类

            support_sequences = support_sequences.float().to(dev)
            # support_seq_labels = support_seq_labels.long().to(dev)

            query_sequences = query_sequences.float().to(dev)
            # query_seq_labels = query_seq_labels.long().to(dev)

            feature, label = maml_DCFAproF_train1(model, support_images, support_sequences,
                                              support_labels,
                                              query_images, query_sequences, query_labels,
                                              3, args, optimizer, 5, is_train=False)

            # 将特征和标签添加到当前SNR下的列表中
            features_snr.append(feature.detach().cpu().numpy())
            labels_snr.append(label.cpu().numpy())  # 需要将标签从GPU移到CPU并转换为numpy数组

    # 将当前SNR下的特征列表添加到所有SNR下的特征列表中
    # 将当前SNR下的特征列表添加到所有SNR下的特征列表中
    all_features.extend(features_snr)
    all_labels.extend(labels_snr)

    # 保存所有SNR下的特征和标签列表到pickle文件中
    with open('./vector_with_maml_for_T-SNE/features.pickle', 'wb') as f:
        pickle.dump(all_features, f)

    with open('./vector_with_maml_for_T-SNE/labels.pickle', 'wb') as f:
        pickle.dump(all_labels, f)
