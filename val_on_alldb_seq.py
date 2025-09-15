import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import RJ_seq_Dataset

from core.helper import get_val_seq_dataset

##  选择模型

from net.maml_1dcnn import maml_1DCNN_train
from net.maml_MS1dcnn import maml_MS1DCNN_train


from args import args, dev


if __name__ == '__main__':

    # model = torch.load('./model_path/best_seq_MS1dcnn.pth')
    model = torch.load('./model_path/best_seq_1dcnn.pth')

    model.eval()  # 将模型设置为评估模式


    params = [p for p in model.parameters() if p.requires_grad]
    start_lr1 = args.outer_lr
    start_lr2 = args.inner_lr
    optimizer = optim.Adam(params, start_lr1)


    # 创建一个信噪比到数据路径的映射
    args.val_data_dirs = {
        -20: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-20_dB/',
        -18: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-18_dB/',
        -16: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-16_dB/',
        -14: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-14_dB/',
        -12: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-12_dB/',
        -10: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-10_dB/',
        -8: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-8_dB/',
        -6: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-6_dB/',
        -4: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-4_dB/',
        -2: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/-2_dB/',
        0: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/0_dB/',
        2: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/2_dB/',
        4: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/4_dB/',
        6: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/6_dB/',
        8: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/8_dB/',
        10: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_seq/10_dB/',
    }

    # args.val_data_dirs = {
    #     -20: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-20_dB/',
    #     -18: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-18_dB/',
    #     -16: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-16_dB/',
    #     -14: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-14_dB/',
    #     -12: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-12_dB/',
    #     -10: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-10_dB/',
    #     -8: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-8_dB/',
    #     -6: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-6_dB/',
    #     -4: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-4_dB/',
    #     -2: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/-2_dB/',
    #     0: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/0_dB/',
    #     2: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/2_dB/',
    #     4: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/4_dB/',
    #     6: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/6_dB/',
    #     8: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/8_dB/',
    #     10: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_seq/10_dB/',
    # }


    snr_levels = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]  # 这里是不同的信噪比级别列表
    valid_loaders = []


    for snr in snr_levels:
        dataset = get_val_seq_dataset(snr, args)
        loader = DataLoader(dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)
        valid_loaders.append(loader)

    # print(valid_loaders)
    total_val_acc =[]

    snr_start = -20
    for i in range(16):
        snr = snr_start
        val_loader = valid_loaders[i]
        val_acc=[]
        for support_sequences, support_labels, query_sequences, query_labels in val_loader:
            # 统一数据类型并转成cuda可读数据类型

            support_sequences = support_sequences.float().to(dev)
            support_labels = support_labels.long().to(dev)

            query_sequences = query_sequences.float().to(dev)
            query_labels = query_labels.long().to(dev)

            ### 用LSTM
            # loss, acc = maml_lstm_train(model, support_sequences, support_labels, query_sequences, query_labels,
            #                               3, args, optimizer, is_train=False)

            ### 用1d-cnn
            loss, acc = maml_1DCNN_train(model, support_sequences, support_labels, query_sequences,
                                                    query_labels, 6, args, optimizer, is_train=False)

            ### 用Multi-Scale 1d-cnn
            # loss, acc = maml_MS1DCNN_train(model, support_sequences, support_labels, query_sequences, query_labels,
            #                                3, args, optimizer, is_train=False)

            ### 用complex-cnn
            # loss, acc = maml_ComplexCNN_train(model, support_sequences, support_labels, query_sequences, query_labels,
            #                              3, args, optimizer, is_train=False)


            val_acc.append(acc)

        snr_start += 2
        print("=> SNR: {:d}   ".format(snr), "=> val_acc: {:.4f}   ".format(np.mean(val_acc)))
        total_val_acc.append(np.mean(val_acc))


    print(total_val_acc)