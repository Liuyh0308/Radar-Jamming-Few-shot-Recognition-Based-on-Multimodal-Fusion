import os

import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np



class ConvBlock(nn.Module):     ## 图片二维卷积Block

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

class ConvBlock1(nn.Module):     ## 序列一维卷积Block

    def __init__(self, in_ch, out_ch):
        super(ConvBlock1, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  ## 3x1卷积  尺寸不变
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化 二分之一降采样

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

class SE_Block(nn.Module):
    def __init__(self, in_channel):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channel, in_channel // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // 16, in_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


class SE_ShrinkBlock(nn.Module):    #### 加入软阈值化 构造残差收缩网络
    def __init__(self, in_channel, shrinkage_threshold=0.5):
        super(SE_ShrinkBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channel, in_channel // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // 16, in_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.shrinkage_threshold = shrinkage_threshold

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        # 应用收缩软阈值化
        x = self.shrinkage(x)

        return x

    def shrinkage(self, x):
        # 软阈值化
        threshold = self.shrinkage_threshold
        return F.relu(x - threshold) - F.relu(-x - threshold)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        self.se = SE_Block(outchannel)


    def forward(self, x):
        out = self.left(x)
        se_out = self.se(out)
        out = out * se_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualBlock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock1, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(outchannel)
        )

        self.se = SE_Block(outchannel)

    def forward(self, x):
        out = self.left(x)
        se_out = self.se(out)
        out = out * se_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out

## 二维卷积
def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, stride=2, padding=3)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    return output


## 一维卷积
def ConvBlockFunction1(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
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

def SEBlockFunction(input, w, b):
    x = F.conv2d(input, w, b)
    output = F.relu(x)
    return output


def SEBlockFunction1(input, w, b):
    x = F.conv2d(input, w, b)
    output = F.sigmoid(x)
    return output


def ResidualBlockFunction(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, stride=1, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    output = F.relu(x)

    return output

def ResidualBlockFunction_1(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, stride=1, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)

    return x


def ResidualBlockFunction1(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, stride=2, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    output = F.relu(x)

    return output

def ResidualBlockFunction1_1(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, stride=1, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)

    return x

def ResidualBlockFunction1_2(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, stride=2, padding=0)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)

    return x



class DCFAproF_Classifier(nn.Module):          ## baseline 四层简单卷积层 可在通道数上做改动 可不变 也可先升维再降维
    def __init__(self, in_ch1, in_ch2, n_way):
        super(DCFAproF_Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch1, 32)       ## 如果要用4层cnn+1层linear imgsize=（28,28）
        self.conv2 = ResidualBlock(32, 32)
        self.conv3 = ResidualBlock(32, 32)
        self.conv4 = ResidualBlock1(32, 64)
        self.conv5 = ResidualBlock(64, 64)
        self.conv6 = ResidualBlock1(64, 128)

        self.fc1 = nn.Linear(512, 256)

        self.conv10a = ConvBlock1(in_ch2, 64)
        self.conv11a = ConvBlock1(64, 64)
        self.conv12a = ConvBlock1(64, 128)
        self.conv13a = ConvBlock1(128, 128)
        self.conv14a = ConvBlock1(128, 128)
        # self.fc2a = nn.Linear(512, 16)

        self.conv10b = ConvBlock0(in_ch2, 64)
        self.conv11b = ConvBlock0(64, 64)
        self.conv12b = ConvBlock0(64, 128)
        self.conv13b = ConvBlock0(128, 128)
        self.conv14b = ConvBlock0(128, 128)
        # self.fc2b = nn.Linear(512, 16)

        self.fcab = nn.Linear(1024, 32)
        # self.fcab = nn.Linear(1024, 64)

        # 分类网络
        self.classifier = nn.Linear(288, n_way)   ## 32+256=288
        # self.classifier = nn.Linear(320, n_way)  ## 64+256=320

        self.fc3 = nn.Linear(256, n_way)
        self.fc4 = nn.Linear(32, n_way)
        # self.fc4 = nn.Linear(64, n_way)

        # self.fc5 = nn.Linear(15, n_way)

        # 定义权重为模型的可训练参数
        # self.weight1 = nn.Parameter(torch.ones(1) / 2)  # 初始化
        # self.weight2 = nn.Parameter(torch.ones(1) / 2.5)  # 初始化
        # self.weight3 = nn.Parameter(torch.ones(1) / 10)
        # self.weights = nn.Parameter(torch.ones(3) / 3)  # 初始化，使得三个权重的总和为1

    def forward(self, x, y):
        ## 图片输入卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        ## 序列输入卷积
        ya = self.conv10a(y)
        ya = self.conv11a(ya)
        ya = self.conv12a(ya)
        ya = self.conv13a(ya)
        ya = self.conv14a(ya)
        ya = F.avg_pool1d(ya, kernel_size=7)
        # ya = ya.view(ya.shape[0], -1)
        # ya = self.fc2a(ya)

        yb = self.conv10b(y)
        yb = self.conv11b(yb)
        yb = self.conv12b(yb)
        yb = self.conv13b(yb)
        yb = self.conv14b(yb)
        yb = F.avg_pool1d(yb, kernel_size=7)
        # yb = yb.view(yb.shape[0], -1)
        # yb = self.fc2b(yb)

        y = torch.concatenate((ya, yb), dim=1)
        y = y.view(y.shape[0], -1)

        y = self.fcab(y)

        z = torch.concat(x, y)
        z = z.view(z.shape[0], -1)

        # 分类
        output = self.classifier(z)
        x = self.fc3(x)
        y = self.fc4(y)

        # combined_pred = torch.cat((output, x, y), dim=1)
        # final_pred = self.fc5(combined_pred)

        # weighted_pred = output * self.weight1 + x * self.weight2 + y * self.weight3
        weighted_pred = output * 0.5 + x * self.weight2 * 0.4 + y * self.weight3 * 0.1

        # return final_pred
        # return output, x, y
        return weighted_pred


    def functional_forward(self, x, y, params):
        x = ConvBlockFunction(x, params['conv1.conv2d.weight'], params[f'conv1.conv2d.bias'],
                              params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))

        x1 = ResidualBlockFunction(x, params[f'conv2.left.0.weight'], params[f'conv2.left.0.bias'],
                                   params.get(f'conv2.left.1.weight'), params.get(f'conv2.left.1.bias'))
        x1 = ResidualBlockFunction_1(x1, params[f'conv2.left.3.weight'], params[f'conv2.left.3.bias'],
                                     params.get(f'conv2.left.4.weight'), params.get(f'conv2.left.4.bias'))
        x1_se = F.avg_pool2d(x1, kernel_size=x1.size(2))
        x1_se = SEBlockFunction(x1_se, params[f'conv2.se.conv1.weight'], params[f'conv2.se.conv1.bias'])
        x1_se = SEBlockFunction1(x1_se, params[f'conv2.se.conv2.weight'], params[f'conv2.se.conv2.bias'])
        x1 = x1 * x1_se
        x1 += x
        x1 = F.relu(x1)

        x2 = ResidualBlockFunction(x1, params[f'conv3.left.0.weight'], params[f'conv3.left.0.bias'],
                                   params.get(f'conv3.left.1.weight'), params.get(f'conv3.left.1.bias'))
        x2 = ResidualBlockFunction_1(x2, params[f'conv3.left.3.weight'], params[f'conv3.left.3.bias'],
                                     params.get(f'conv3.left.4.weight'), params.get(f'conv3.left.4.bias'))
        x2_se = F.avg_pool2d(x2, kernel_size=x2.size(2))
        x2_se = SEBlockFunction(x2_se, params[f'conv3.se.conv1.weight'], params[f'conv3.se.conv1.bias'])
        x2_se = SEBlockFunction1(x2_se, params[f'conv3.se.conv2.weight'], params[f'conv3.se.conv2.bias'])
        x2 = x2 * x2_se
        x2 += x1
        x2 = F.relu(x2)

        x3_0 = ResidualBlockFunction1_2(x2, params[f'conv4.shortcut.0.weight'], params[f'conv4.shortcut.0.bias'],
                                        params.get(f'conv4.shortcut.1.weight'), params.get(f'conv4.shortcut.1.bias'))
        x3 = ResidualBlockFunction1(x2, params[f'conv4.left.0.weight'], params[f'conv4.left.0.bias'],
                                    params.get(f'conv4.left.1.weight'), params.get(f'conv4.left.1.bias'))

        x3 = ResidualBlockFunction1_1(x3, params[f'conv4.left.3.weight'], params[f'conv4.left.3.bias'],
                                      params.get(f'conv4.left.4.weight'), params.get(f'conv4.left.4.bias'))
        x3_se = F.avg_pool2d(x3, kernel_size=x3.size(2))
        x3_se = SEBlockFunction(x3_se, params[f'conv4.se.conv1.weight'], params[f'conv4.se.conv1.bias'])
        x3_se = SEBlockFunction1(x3_se, params[f'conv4.se.conv2.weight'], params[f'conv4.se.conv2.bias'])
        x3 = x3 * x3_se
        x3 += x3_0
        x3 = F.relu(x3)

        x4 = ResidualBlockFunction(x3, params[f'conv5.left.0.weight'], params[f'conv5.left.0.bias'],
                                   params.get(f'conv5.left.1.weight'), params.get(f'conv5.left.1.bias'))
        x4 = ResidualBlockFunction_1(x4, params[f'conv5.left.3.weight'], params[f'conv5.left.3.bias'],
                                     params.get(f'conv5.left.4.weight'), params.get(f'conv5.left.4.bias'))
        x4_se = F.avg_pool2d(x4, kernel_size=x4.size(2))
        x4_se = SEBlockFunction(x4_se, params[f'conv5.se.conv1.weight'], params[f'conv5.se.conv1.bias'])
        x4_se = SEBlockFunction1(x4_se, params[f'conv5.se.conv2.weight'], params[f'conv5.se.conv2.bias'])
        x4 = x4 * x4_se
        x4 += x3
        x4 = F.relu(x4)

        x5_0 = ResidualBlockFunction1_2(x4, params[f'conv6.shortcut.0.weight'], params[f'conv6.shortcut.0.bias'],
                                        params.get(f'conv6.shortcut.1.weight'), params.get(f'conv6.shortcut.1.bias'))
        x5 = ResidualBlockFunction1(x4, params[f'conv6.left.0.weight'], params[f'conv6.left.0.bias'],
                                    params.get(f'conv6.left.1.weight'), params.get(f'conv6.left.1.bias'))
        x5 = ResidualBlockFunction1_1(x5, params[f'conv6.left.3.weight'], params[f'conv6.left.3.bias'],
                                      params.get(f'conv6.left.4.weight'), params.get(f'conv6.left.4.bias'))
        x5_se = F.avg_pool2d(x5, kernel_size=x5.size(2))
        x5_se = SEBlockFunction(x5_se, params[f'conv6.se.conv1.weight'], params[f'conv6.se.conv1.bias'])
        x5_se = SEBlockFunction1(x5_se, params[f'conv6.se.conv2.weight'], params[f'conv6.se.conv2.bias'])
        x5 = x5 * x5_se
        x5 += x5_0
        x5 = F.relu(x5)

        x = F.avg_pool2d(x5, kernel_size=7)
        x = x.view(x.shape[0], -1)

        x = F.linear(x, params[f'fc1.weight'], params[f'fc1.bias'])



        ya = ConvBlockFunction1(y, params[f'conv10a.conv1d.weight'], params[f'conv10a.conv1d.bias'],
                              params.get(f'conv10a.bn.weight'), params.get(f'conv10a.bn.bias'))
        ya = ConvBlockFunction1(ya, params[f'conv11a.conv1d.weight'], params[f'conv11a.conv1d.bias'],
                              params.get(f'conv11a.bn.weight'), params.get(f'conv11a.bn.bias'))
        ya = ConvBlockFunction1(ya, params[f'conv12a.conv1d.weight'], params[f'conv12a.conv1d.bias'],
                               params.get(f'conv12a.bn.weight'), params.get(f'conv12a.bn.bias'))
        ya = ConvBlockFunction1(ya, params[f'conv13a.conv1d.weight'], params[f'conv13a.conv1d.bias'],
                               params.get(f'conv13a.bn.weight'), params.get(f'conv13a.bn.bias'))
        ya = ConvBlockFunction1(ya, params[f'conv14a.conv1d.weight'], params[f'conv14a.conv1d.bias'],
                               params.get(f'conv14a.bn.weight'), params.get(f'conv14a.bn.bias'))

        ya = F.avg_pool1d(ya, kernel_size=7)

        yb = ConvBlockFunction0(y, params[f'conv10b.conv1d.weight'], params[f'conv10b.conv1d.bias'],
                               params.get(f'conv10b.bn.weight'), params.get(f'conv10b.bn.bias'))
        yb = ConvBlockFunction0(yb, params[f'conv11b.conv1d.weight'], params[f'conv11b.conv1d.bias'],
                               params.get(f'conv11b.bn.weight'), params.get(f'conv11b.bn.bias'))
        yb = ConvBlockFunction0(yb, params[f'conv12b.conv1d.weight'], params[f'conv12b.conv1d.bias'],
                               params.get(f'conv12b.bn.weight'), params.get(f'conv12b.bn.bias'))
        yb = ConvBlockFunction0(yb, params[f'conv13b.conv1d.weight'], params[f'conv13b.conv1d.bias'],
                               params.get(f'conv13b.bn.weight'), params.get(f'conv13b.bn.bias'))
        yb = ConvBlockFunction0(yb, params[f'conv14b.conv1d.weight'], params[f'conv14b.conv1d.bias'],
                               params.get(f'conv14b.bn.weight'), params.get(f'conv14b.bn.bias'))

        yb = F.avg_pool1d(yb, kernel_size=7)

        y = torch.concatenate((ya, yb), dim=1)

        y = y.view(y.shape[0], -1)
        # y = F.linear(y, params[f'fc2.weight'], params[f'fc2.bias'])

        y = F.linear(y, params[f'fcab.weight'], params[f'fcab.bias'])

        z = torch.concat((x, y), dim=1)
        z = z.view(z.shape[0], -1)

        fused_feature = F.linear(z, params['classifier.weight'], params['classifier.bias'])

        x = F.linear(x, params[f'fc3.weight'], params[f'fc3.bias'])
        y = F.linear(y, params[f'fc4.weight'], params[f'fc4.bias'])

        ## 学习融合方法
        # combined_pred = torch.cat((fused_feature, x, y), dim=1)
        # final_pred = F.linear(combined_pred, params[f'fc5.weight'], params[f'fc5.bias'])

        ## 自适应加权平均方法
        # weighted_pred = fused_feature * params[f'weight1'] + x * params[f'weight2'] + y * params[f'weight3']
        # weighted_pred = fused_feature * 0.5 + x * 0.4 + y * 0.1

        return fused_feature, x, y
        # return x5

        # return final_pred
        # return weighted_pred




def maml_DCFAproF_train(model, support_images, support_sequences, support_labels,
                      query_images, query_sequences, query_labels, inner_step, args, optimizer, n_way, is_train=True):
    """
    Train the model using MAML method.
    Args:
        model: Any model
        support_images: several task support images
        support_labels: several  support labels
        query_images: several query images
        query_labels: several query labels
        inner_step: support data training step
        args: ArgumentParser
        optimizer: optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []

    ##混淆矩阵初始化定义
    n_classes = n_way  # 确保这与您的类别数匹配
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)


    # Initialize a dictionary to track accuracy per class
    acc_per_class = {label.item(): [] for label in torch.unique(query_labels)}

    for support_image, support_sequence, support_label, query_image, query_sequence, query_label \
            in zip(support_images, support_sequences, support_labels, query_images, query_sequences, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())

        for _ in range(inner_step):
            # 普通法更新权重
            # support_logit = model.functional_forward(support_image, support_sequence, fast_weights)
            # support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)

            ## 采用多模态协同构架，loss计算的不单单只是融合后的多模态特征计算得到的loss，并且还叠加了两个单模态的loss，均使用交叉熵进行loss的计算
            support_logit, support_logit_x, support_logit_y = model.functional_forward(support_image, support_sequence, fast_weights)

            support_loss = (nn.CrossEntropyLoss().cuda()(support_logit, support_label)+nn.CrossEntropyLoss().cuda()(support_logit_x, support_label) +
                            nn.CrossEntropyLoss().cuda()(support_logit_y, support_label))

            # support_loss = (nn.CrossEntropyLoss().cuda()(support_logit, support_label) + 7 / 8 * (
            #     nn.CrossEntropyLoss().cuda()(support_logit_x, support_logit)) + 1 / 8 * (nn.CrossEntropyLoss().cuda()(support_logit_y, support_logit)))

            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - args.inner_lr * grads)
                                                   for ((name, param), grads) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        # query_logit = model.functional_forward(query_image, query_sequence, fast_weights)
        query_logit, query_logit_x, query_logit_y = model.functional_forward(query_image, query_sequence, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        # query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        #
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)
        ## 采用多模态协同构架，loss计算的不单单只是融合后的多模态特征计算得到的loss，并且还叠加了两个单模态的loss，均使用交叉熵进行loss的计算
        query_loss = (nn.CrossEntropyLoss().cuda()(query_logit, query_label) + nn.CrossEntropyLoss().cuda()(query_logit_x, query_label) +
                      nn.CrossEntropyLoss().cuda()(query_logit_y, query_label))

        # query_loss = ((nn.CrossEntropyLoss().cuda()(query_logit, query_label)) + 7 / 8 * (nn.CrossEntropyLoss().cuda()(query_logit_x, query_label))
        #               + 1 / 8 * (nn.CrossEntropyLoss().cuda()(query_logit_y, query_label)))


        # 更新混淆矩阵，验证用，训练需注释
        for t, p in zip(query_label.view(-1), query_prediction.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1



       ## 用于val时计算各个类别的准确率（训练时要注释掉，减少变量存储，节省内存）
        if is_train == False:
            # Calculate accuracy for each class
            for label in acc_per_class.keys():
                indices = (query_label == label)
                if indices.any():
                    acc = torch.eq(query_prediction[indices], query_label[indices]).float().mean().item()
                    acc_per_class[label].append(acc)


        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())

    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    # 计算每个类别的平均正确率  验证用
    acc_per_class = {label: np.mean(accs) if accs else 0 for label, accs in acc_per_class.items()}

    if is_train:
        meta_loss.backward()
        optimizer.step()

    # return meta_loss, meta_acc   ##训练用
    # return meta_loss, meta_acc, acc_per_class
    return meta_loss, meta_acc, acc_per_class, confusion_matrix








def maml_DCFAproF_train1(model, support_images, support_sequences, support_labels,
                      query_images, query_sequences, query_labels, inner_step, args, optimizer, n_way, is_train=True):
    """
    Train the model using MAML method.
    Args:
        model: Any model
        support_images: several task support images
        support_labels: several  support labels
        query_images: several query images
        query_labels: several query labels
        inner_step: support data training step
        args: ArgumentParser
        optimizer: optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []


    for support_image, support_sequence, support_label, query_image, query_sequence, query_label \
            in zip(support_images, support_sequences, support_labels, query_images, query_sequences, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())
        feature = model.functional_forward(support_image, support_sequence, fast_weights)




    # return meta_loss, meta_acc, acc_per_class
    return feature, support_label
