import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

class ConvBlock0(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock0, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, 5, stride=1, padding=2)
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
    x = F.conv1d(input, w, b, stride=1, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool1d(x, kernel_size=2, stride=2)

    return output


def ConvBlockFunction0(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv1d(input, w, b, stride=1, padding=2)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool1d(x, kernel_size=2, stride=2)

    return output

class Classifier_MS1DCNN(nn.Module):          ## baseline 四层简单卷积层 可在通道数上做改动 可不变 也可先升维再降维
    def __init__(self, in_ch, n_way):
        super(Classifier_MS1DCNN, self).__init__()

   ### 和不使用MAML方法的普通深度学习对比效果
        self.conv1a = ConvBlock(in_ch, 64)
        self.conv2a = ConvBlock(64, 64)
        self.conv3a = ConvBlock(64, 128)
        self.conv4a = ConvBlock(128, 128)
        self.conv5a = ConvBlock(128, 128)
        self.conv6a = ConvBlock(128, 128)
        self.conv7a = ConvBlock(128, 256)
        # self.conv8a = ConvBlock(256, 256)
        # self.conv9a = ConvBlock(256, 256)
        # self.conv10a = ConvBlock(256, 256)

        self.conv1b = ConvBlock0(in_ch, 64)
        self.conv2b = ConvBlock0(64, 64)
        self.conv3b = ConvBlock0(64, 128)
        self.conv4b = ConvBlock0(128, 128)
        self.conv5b = ConvBlock0(128, 128)
        self.conv6b = ConvBlock0(128, 128)
        self.conv7b = ConvBlock0(128, 256)
        # self.conv8b = ConvBlock0(256, 256)
        # self.conv9b = ConvBlock0(256, 256)
        # self.conv10b = ConvBlock0(256, 256)

        self.logits1 = nn.Linear(256 * 2, 128)
        self.logits2 = nn.Linear(128, n_way)
        # self.logits = nn.Linear(128 * 2, n_way)

    def forward(self, x):
        xa = self.conv1a(x)
        xa = self.conv2a(xa)
        xa = self.conv3a(xa)
        xa = self.conv4a(xa)
        xa = self.conv5a(xa)
        xa = self.conv6a(xa)
        xa = self.conv7a(xa)
        xa = F.avg_pool1d(xa, kernel_size=7)
        # xa = self.conv8a(xa)
        # xa = self.conv9a(xa)
        # xa = self.conv10a(xa)

        xb = self.conv1b(x)
        xb = self.conv2b(xb)
        xb = self.conv3b(xb)
        xb = self.conv4b(xb)
        xb = self.conv5b(xb)
        xb = self.conv6b(xb)
        xb = self.conv7b(xb)
        xb = F.avg_pool1d(xb, kernel_size=7)
        # xb = self.conv8b(xb)
        # xb = self.conv9b(xb)
        # xb = self.conv10b(xb)

        x = torch.concatenate((xa, xb), dim=1)

        x = x.view(x.shape[0], -1)
        x = self.logits1(x)
        x = self.logits2(x)

        return x


    def functional_forward(self, x, params):
        xa = ConvBlockFunction(x, params[f'conv1a.conv1d.weight'], params[f'conv1a.conv1d.bias'],
                              params.get(f'conv1a.bn.weight'), params.get(f'conv1a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv2a.conv1d.weight'], params[f'conv2a.conv1d.bias'],
                              params.get(f'conv2a.bn.weight'), params.get(f'conv2a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv3a.conv1d.weight'], params[f'conv3a.conv1d.bias'],
                              params.get(f'conv3a.bn.weight'), params.get(f'conv3a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv4a.conv1d.weight'], params[f'conv4a.conv1d.bias'],
                              params.get(f'conv4a.bn.weight'), params.get(f'conv4a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv5a.conv1d.weight'], params[f'conv5a.conv1d.bias'],
                              params.get(f'conv5a.bn.weight'), params.get(f'conv5a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv6a.conv1d.weight'], params[f'conv6a.conv1d.bias'],
                              params.get(f'conv6a.bn.weight'), params.get(f'conv6a.bn.bias'))
        xa = ConvBlockFunction(xa, params[f'conv7a.conv1d.weight'], params[f'conv7a.conv1d.bias'],
                              params.get(f'conv7a.bn.weight'), params.get(f'conv7a.bn.bias'))
        xa = F.avg_pool1d(xa, kernel_size=7)
        # xa = ConvBlockFunction(xa, params[f'conv8a.conv1d.weight'], params[f'conv8a.conv1d.bias'],
        #                       params.get(f'conv8a.bn.weight'), params.get(f'conv8a.bn.bias'))
        # xa = ConvBlockFunction(xa, params[f'conv9a.conv1d.weight'], params[f'conv9a.conv1d.bias'],
        #                       params.get(f'conv9a.bn.weight'), params.get(f'conv9a.bn.bias'))
        # xa = ConvBlockFunction(xa, params[f'conv10a.conv1d.weight'], params[f'conv10a.conv1d.bias'],
        #                       params.get(f'conv10a.bn.weight'), params.get(f'conv10a.bn.bias'))

        xb = ConvBlockFunction0(x, params[f'conv1b.conv1d.weight'], params[f'conv1b.conv1d.bias'],
                               params.get(f'conv1b.bn.weight'), params.get(f'conv1b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv2b.conv1d.weight'], params[f'conv2b.conv1d.bias'],
                               params.get(f'conv2b.bn.weight'), params.get(f'conv2b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv3b.conv1d.weight'], params[f'conv3b.conv1d.bias'],
                               params.get(f'conv3b.bn.weight'), params.get(f'conv3b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv4b.conv1d.weight'], params[f'conv4b.conv1d.bias'],
                               params.get(f'conv4b.bn.weight'), params.get(f'conv4b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv5b.conv1d.weight'], params[f'conv5b.conv1d.bias'],
                               params.get(f'conv5b.bn.weight'), params.get(f'conv5b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv6b.conv1d.weight'], params[f'conv6b.conv1d.bias'],
                               params.get(f'conv6b.bn.weight'), params.get(f'conv6b.bn.bias'))
        xb = ConvBlockFunction0(xb, params[f'conv7b.conv1d.weight'], params[f'conv7b.conv1d.bias'],
                               params.get(f'conv7b.bn.weight'), params.get(f'conv7b.bn.bias'))
        xb = F.avg_pool1d(xb, kernel_size=7)
        # xb = ConvBlockFunction0(xb, params[f'conv8b.conv1d.weight'], params[f'conv8b.conv1d.bias'],
        #                        params.get(f'conv8b.bn.weight'), params.get(f'conv8b.bn.bias'))
        # xb = ConvBlockFunction0(xb, params[f'conv9b.conv1d.weight'], params[f'conv9b.conv1d.bias'],
        #                        params.get(f'conv9b.bn.weight'), params.get(f'conv9b.bn.bias'))
        # xb = ConvBlockFunction0(xb, params[f'conv10b.conv1d.weight'], params[f'conv10b.conv1d.bias'],
        #                        params.get(f'conv10b.bn.weight'), params.get(f'conv10b.bn.bias'))

        x = torch.concatenate((xa, xb), dim=1)
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits1.weight'], params['logits1.bias'])
        x = F.linear(x, params['logits2.weight'], params['logits2.bias'])

        return x





def maml_MS1DCNN_train(model, support_sequences, support_labels, query_sequences, query_labels, inner_step, args, optimizer, is_train=True):

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
    confusion = confusion_matrix(y_true, y_pred)


    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc ## ,confusion
