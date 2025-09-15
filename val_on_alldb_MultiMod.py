import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import RJ_fusion_Dataset

from core.helper import get_val_fusion_dataset, compute_metrics

##  选择模型
from net.maml_fusion_net import maml_fusion_train
from net.maml_lstm_res18_fusionnet import maml_lstm_res18_fusion_train
from net.maml_DCFAnet import maml_DCFA_train
from net.maml_DCFAproF import maml_DCFAproF_train, maml_DCFAproF_train1

from args import args, dev


if __name__ == '__main__':

    model = torch.load('./model_path/best_fusion_DCFAproF.pth')       ## 效果较好
    # model = torch.load('./model_path/best_fusion_DCFAproF1.pth')

    model.eval()  # 将模型设置为评估模式


    params = [p for p in model.parameters() if p.requires_grad]
    start_lr1 = args.outer_lr
    start_lr2 = args.inner_lr
    optimizer = optim.Adam(params, start_lr1)


    # 创建一个信噪比到数据路径的映射

    # 验证集Z
    # args.val_data_img_dirs = {
    #     -20: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-20_dB/',
    #     -18: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-18_dB/',
    #     -16: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-16_dB/',
    #     -14: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-14_dB/',
    #     -12: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-12_dB/',
    #     -10: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-10_dB/',
    #     -8: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-8_dB/',
    #     -6: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-6_dB/',
    #     -4: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-4_dB/',
    #     -2: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/-2_dB/',
    #     0: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/0_dB/',
    #     2: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/2_dB/',
    #     4: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/4_dB/',
    #     6: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/6_dB/',
    #     8: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/8_dB/',
    #     10: './dataset/Radar_Jamming_Signal_Dataset/Test_data_Z/dataset_img/10_dB/',
    # }
    #
    # args.val_data_seq_dirs = {
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

  ## 普通验证集
    args.val_data_img_dirs = {
        -20: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-20_dB/',
        -18: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-18_dB/',
        -16: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-16_dB/',
        -14: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-14_dB/',
        -12: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-12_dB/',
        -10: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-10_dB/',
        -8: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-8_dB/',
        -6: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-6_dB/',
        -4: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-4_dB/',
        -2: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/-2_dB/',
        0: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/0_dB/',
        2: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/2_dB/',
        4: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/4_dB/',
        6: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/6_dB/',
        8: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/8_dB/',
        10: './dataset/Radar_Jamming_Signal_Dataset/Test_data/dataset_img/10_dB/',
    }

    args.val_data_seq_dirs = {
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




    snr_levels = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]  # 这里是不同的信噪比级别列表
    valid_loaders = []

    class_accuracies = {snr: {} for snr in snr_levels}  # To store class-wise accuracies for each SNR


    for snr in snr_levels:
        dataset = get_val_fusion_dataset(snr, args)
        loader = DataLoader(dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)
        valid_loaders.append(loader)

    # print(valid_loaders)   ## 测试valid_loader是否创建成功

    total_val_acc =[]

    ## 初始化全局混淆矩阵
    confusion_matrices = {snr: np.zeros((args.n_way, args.n_way), dtype=int) for snr in snr_levels}

    snr_start = -20
    for i in range(16):
        snr = snr_start
        val_loader = valid_loaders[i]
        val_acc=[]

        acc_per_class = {}  # Temporary storage for class accuracies at this SNR

        for support_images, support_img_labels, query_images, query_img_labels, \
            support_sequences, support_seq_labels, query_sequences, query_seq_labels in val_loader:

            # 得到变量
            support_images = support_images.float().to(dev)
            # support_img_labels = support_img_labels.long().to(dev)
            support_labels = support_img_labels.long().to(dev)  ## 对图片与序列support集与query集都只返回一个总label，而不去区分support_img_label or query_img_label之类

            query_images = query_images.float().to(dev)
            # query_img_labels = query_img_labels.long().to(dev)
            query_labels = query_img_labels.long().to(dev)      ## 对图片与序列support集与query集都只返回一个总label，而不去区分support_img_label or query_img_label之类

            support_sequences = support_sequences.float().to(dev)
            # support_seq_labels = support_seq_labels.long().to(dev)

            query_sequences = query_sequences.float().to(dev)
            # query_seq_labels = query_seq_labels.long().to(dev)


            ### 用多模态网络作为backbone训练
            # loss, acc = maml_fusion_train(model, support_images, support_sequences, support_labels,
            #                               query_images, query_sequences, query_labels, 3, args, optimizer, is_train=False)

            # loss, acc = maml_lstm_res18_fusion_train(model, support_images, support_sequences, support_labels,
            #                               query_images, query_sequences, query_labels, 3, args, optimizer, is_train=False)

            # loss, acc = maml_DCFA_train(model, support_images, support_sequences, support_labels,
            #                             query_images, query_sequences, query_labels, 3, args, optimizer, is_train=False)

            loss, acc, acc_by_class, confusion_matrix = maml_DCFAproF_train(model, support_images, support_sequences, support_labels,
                                            query_images, query_sequences, query_labels, 3, args, optimizer, 5, is_train=False)



            val_acc.append(acc)

            ## 在每个batch后更新混淆矩阵
            confusion_matrices[snr] += confusion_matrix     # 累加混淆矩阵

            for class_id, class_acc in acc_by_class.items():
                if class_id not in acc_per_class:
                    acc_per_class[class_id] = []
                acc_per_class[class_id].append(class_acc)


        snr_start += 2

        ## 输出每个SNR的混淆矩阵
        print(f"Confusion Matrix for SNR {snr}dB:")
        print(confusion_matrices[snr])

        ## 计算并输出每个SNR的混淆矩阵及其指标
        accuracy, precision, recall, f1_scores = compute_metrics(confusion_matrices[snr])
        print(f"Metrics: Acc: {accuracy:.4f}, Precision: {precision}, Recall: {recall}, F1: {f1_scores}")

        # 输出每个SNR下的val_acc
        print("=> SNR: {:d}   ".format(snr), "=> val_acc: {:.4f}   ".format(np.mean(val_acc)))
        total_val_acc.append(np.mean(val_acc))

        class_accuracies[snr] = {class_id: np.mean(class_accs) for class_id, class_accs in acc_per_class.items()}

    ## 输出每个SNR下的每一类信号的acc
    for snr, accuracies in class_accuracies.items():
        print(f"\nClass-wise accuracies for SNR: {snr}dB:")
        for class_id, acc in accuracies.items():
            print(f"Class {class_id}: {acc:.4f}")


    print(total_val_acc)