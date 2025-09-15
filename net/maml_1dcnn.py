import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, 3 , padding=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv1d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool1d(x, kernel_size=2, stride=2)

    return output


class Classifier_1DCNN(nn.Module):          ## baseline 四层简单卷积层 可在通道数上做改动 可不变 也可先升维再降维
    def __init__(self, in_ch, n_way):
        super(Classifier_1DCNN, self).__init__()
        # self.conv1 = ConvBlock(in_ch, 64)       ## 如果要用4层cnn+1层linear imgsize=（28,28）
        # self.conv2 = ConvBlock(64, 64)
        # self.conv3 = ConvBlock(64, 64)
        # self.conv4 = ConvBlock(64, 64)
        # self.logits = nn.Linear(64, n_way)

   ### 和不使用MAML方法的普通深度学习对比效果
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 512)
        # self.conv8 = ConvBlock(256, 256)
        # self.conv9 = ConvBlock(256, 256)
        # self.conv10 = ConvBlock(256, 512)
        self.avgpool = nn.AvgPool1d(kernel_size=7)
        self.logits1 = nn.Linear(512, 128)
        self.logits2 = nn.Linear(128, n_way)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        # x = self.conv10(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.logits1(x)
        x = self.logits2(x)

        return x


    def functional_forward(self, x, params):
        x = ConvBlockFunction(x, params[f'conv1.conv1d.weight'], params[f'conv1.conv1d.bias'],
                              params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv2.conv1d.weight'], params[f'conv2.conv1d.bias'],
                              params.get(f'conv2.bn.weight'), params.get(f'conv2.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv3.conv1d.weight'], params[f'conv3.conv1d.bias'],
                              params.get(f'conv3.bn.weight'), params.get(f'conv3.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv4.conv1d.weight'], params[f'conv4.conv1d.bias'],
                              params.get(f'conv4.bn.weight'), params.get(f'conv4.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv5.conv1d.weight'], params[f'conv5.conv1d.bias'],
                              params.get(f'conv5.bn.weight'), params.get(f'conv5.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv6.conv1d.weight'], params[f'conv6.conv1d.bias'],
                              params.get(f'conv6.bn.weight'), params.get(f'conv6.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv7.conv1d.weight'], params[f'conv7.conv1d.bias'],
                              params.get(f'conv7.bn.weight'), params.get(f'conv7.bn.bias'))
        x = F.avg_pool1d(x, kernel_size=7)
        # x = ConvBlockFunction(x, params[f'conv8.conv1d.weight'], params[f'conv8.conv1d.bias'],
        #                       params.get(f'conv8.bn.weight'), params.get(f'conv8.bn.bias'))
        # x = ConvBlockFunction(x, params[f'conv9.conv1d.weight'], params[f'conv9.conv1d.bias'],
        #                       params.get(f'conv9.bn.weight'), params.get(f'conv9.bn.bias'))
        # x = ConvBlockFunction(x, params[f'conv10.conv1d.weight'], params[f'conv10.conv1d.bias'],
        #                       params.get(f'conv10.bn.weight'), params.get(f'conv10.bn.bias'))


        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits1.weight'], params['logits1.bias'])
        x = F.linear(x, params['logits2.weight'], params['logits2.bias'])

        return x





def maml_1DCNN_train(model, support_sequences, support_labels, query_sequences, query_labels, inner_step, args, optimizer, is_train=True):

    meta_loss = []
    meta_acc = []

    y_true = []  # 用于存储真实标签
    y_pred = []  # 用于存储预测标签

    for support_sequence, support_label, query_sequence, query_label in zip(support_sequences, support_labels, query_sequences, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())

        for _ in range(inner_step):
            # Update weight
            support_logit = model.functional_forward(support_sequence, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - args.inner_lr * grads)
                                                   for ((name, param), grads) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        query_logit = model.functional_forward(query_sequence, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]


        y_true.extend(query_label.cpu().numpy())
        y_pred.extend(query_prediction.cpu().numpy())


        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())


    # 计算训练集混淆矩阵
    # confusion = confusion_matrix(y_true, y_pred)


    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc
