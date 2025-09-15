import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np



class ConvBlock(nn.Module):     ## 图片二维卷积Block

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)  ## 3x3卷积  尺寸不变
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化 二分之一降采样

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x

class ConvBlock1(nn.Module):     ## 序列一维卷积Block

    def __init__(self, in_ch, out_ch):
        super(ConvBlock1, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, 3 , padding=1)  ## 3x1卷积  尺寸不变
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化 二分之一降采样

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


class ConvBlock2(nn.Module):     ## 序列一维卷积Block

    def __init__(self, in_ch, out_ch):
        super(ConvBlock2, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, 3 , padding=1)  ## 3x1卷积  尺寸不变
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)  # 最大池化

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


def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, stride=2, padding=3)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    return output


def ConvBlockFunction1(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv1d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool1d(x, kernel_size=2, stride=2)

    return output

def ConvBlockFunction2(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv1d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool1d(x, kernel_size=2, stride=2, padding=1)

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


class LSTM_Res18_Fusion_Classifier(nn.Module):          ## baseline 四层简单卷积层 可在通道数上做改动 可不变 也可先升维再降维
    def __init__(self, in_ch1, in_ch2, n_way):
        super(LSTM_Res18_Fusion_Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch1, 64)       ## 如果要用4层cnn+1层linear imgsize=（28,28）
        self.conv2 = ResidualBlock(64, 64)
        self.conv3 = ResidualBlock(64, 64)
        self.conv4 = ResidualBlock1(64, 128)
        self.conv5 = ResidualBlock(128, 128)
        self.conv6 = ResidualBlock1(128, 256)
        self.conv7 = ResidualBlock(256, 256)
        self.conv8 = ResidualBlock1(256, 512)
        self.conv9 = ResidualBlock(512, 512)

        self.conv10 = ConvBlock1(in_ch2, 64)
        self.conv11 = ConvBlock1(64, 64)
        self.conv12 = ConvBlock1(64, 64)
        self.conv13 = ConvBlock1(64, 128)
        self.conv14 = ConvBlock1(128, 128)
        self.conv15 = ConvBlock1(128, 256)
        self.conv16 = ConvBlock1(256, 256)
        self.conv17 = ConvBlock1(256, 512)
        self.conv18 = ConvBlock1(512, 512)
        self.conv19 = ConvBlock1(512, 512)


        # self.lstm1 = nn.LSTM()
        # self.conv20 = ConvBlock2(1024, 1024)
        # self.conv21 = ConvBlock2(1024, 1024)

        self.logits1 = nn.Linear(1024, 1024)
        self.logits2 = nn.Linear(1024, n_way)


    def forward(self, x, y):
        ## 图片输入卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x)

        ## 序列输入卷积
        y = self.conv10(y)
        y = self.conv11(y)
        y = self.conv12(y)
        y = self.conv13(y)
        y = self.conv14(y)
        y = self.conv15(y)
        y = self.conv16(y)
        y = self.conv17(y)
        y = self.conv18(y)
        y = self.conv19(y)

        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        z = torch.concat(x, y)
        # z = self.conv20(z)
        # z = self.conv21(z)

        z = self.logits1(z)
        z = self.logits2(z)

        return z


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

        x6 = ResidualBlockFunction(x5, params[f'conv7.left.0.weight'], params[f'conv7.left.0.bias'],
                                   params.get(f'conv7.left.1.weight'), params.get(f'conv7.left.1.bias'))
        x6 = ResidualBlockFunction_1(x6, params[f'conv7.left.3.weight'], params[f'conv7.left.3.bias'],
                                     params.get(f'conv7.left.4.weight'), params.get(f'conv7.left.4.bias'))
        x6_se = F.avg_pool2d(x6, kernel_size=x6.size(2))
        x6_se = SEBlockFunction(x6_se, params[f'conv7.se.conv1.weight'], params[f'conv7.se.conv1.bias'])
        x6_se = SEBlockFunction1(x6_se, params[f'conv7.se.conv2.weight'], params[f'conv7.se.conv2.bias'])
        x6 = x6 * x6_se
        x6 += x5
        x6 = F.relu(x6)

        x7_0 = ResidualBlockFunction1_2(x6, params[f'conv8.shortcut.0.weight'], params[f'conv8.shortcut.0.bias'],
                                        params.get(f'conv8.shortcut.1.weight'), params.get(f'conv8.shortcut.1.bias'))
        x7 = ResidualBlockFunction1(x6, params[f'conv8.left.0.weight'], params[f'conv8.left.0.bias'],
                                    params.get(f'conv8.left.1.weight'), params.get(f'conv8.left.1.bias'))
        x7 = ResidualBlockFunction1_1(x7, params[f'conv8.left.3.weight'], params[f'conv8.left.3.bias'],
                                      params.get(f'conv8.left.4.weight'), params.get(f'conv8.left.4.bias'))
        x7_se = F.avg_pool2d(x7, kernel_size=x7.size(2))
        x7_se = SEBlockFunction(x7_se, params[f'conv8.se.conv1.weight'], params[f'conv8.se.conv1.bias'])
        x7_se = SEBlockFunction1(x7_se, params[f'conv8.se.conv2.weight'], params[f'conv8.se.conv2.bias'])
        x7 = x7 * x7_se
        x7 += x7_0
        x7 = F.relu(x7)

        x8 = ResidualBlockFunction(x7, params[f'conv9.left.0.weight'], params[f'conv9.left.0.bias'],
                                   params.get(f'conv9.left.1.weight'), params.get(f'conv9.left.1.bias'))
        x8 = ResidualBlockFunction_1(x8, params[f'conv9.left.3.weight'], params[f'conv9.left.3.bias'],
                                     params.get(f'conv9.left.4.weight'), params.get(f'conv9.left.4.bias'))
        x8_se = F.avg_pool2d(x8, kernel_size=x8.size(2))
        x8_se = SEBlockFunction(x8_se, params[f'conv9.se.conv1.weight'], params[f'conv9.se.conv1.bias'])
        x8_se = SEBlockFunction1(x8_se, params[f'conv9.se.conv2.weight'], params[f'conv9.se.conv2.bias'])
        x8 = x8 * x8_se
        x8 += x7
        x8 = F.relu(x8)

        x9 = F.avg_pool2d(x8, kernel_size=7)


        y = ConvBlockFunction1(y, params[f'conv10.conv1d.weight'], params[f'conv10.conv1d.bias'],
                              params.get(f'conv10.bn.weight'), params.get(f'conv10.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv11.conv1d.weight'], params[f'conv11.conv1d.bias'],
                              params.get(f'conv11.bn.weight'), params.get(f'conv11.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv12.conv1d.weight'], params[f'conv12.conv1d.bias'],
                               params.get(f'conv12.bn.weight'), params.get(f'conv12.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv13.conv1d.weight'], params[f'conv13.conv1d.bias'],
                               params.get(f'conv13.bn.weight'), params.get(f'conv13.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv14.conv1d.weight'], params[f'conv14.conv1d.bias'],
                               params.get(f'conv14.bn.weight'), params.get(f'conv14.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv15.conv1d.weight'], params[f'conv15.conv1d.bias'],
                               params.get(f'conv15.bn.weight'), params.get(f'conv15.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv16.conv1d.weight'], params[f'conv16.conv1d.bias'],
                               params.get(f'conv16.bn.weight'), params.get(f'conv16.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv17.conv1d.weight'], params[f'conv17.conv1d.bias'],
                               params.get(f'conv17.bn.weight'), params.get(f'conv17.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv18.conv1d.weight'], params[f'conv18.conv1d.bias'],
                               params.get(f'conv18.bn.weight'), params.get(f'conv18.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv19.conv1d.weight'], params[f'conv19.conv1d.bias'],
                               params.get(f'conv19.bn.weight'), params.get(f'conv19.bn.bias'))

        x9 = x9.view(x9.shape[0], x9.shape[1], -1)
        # y = y.view(y.shape[0], -1)

        z = torch.concat((x9, y), dim=1)
        # z = ConvBlockFunction2(z, params[f'conv20.conv1d.weight'], params[f'conv20.conv1d.bias'],
        #                        params.get(f'conv20.bn.weight'), params.get(f'conv20.bn.bias'))
        # z = ConvBlockFunction2(z, params[f'conv21.conv1d.weight'], params[f'conv21.conv1d.bias'],
        #                        params.get(f'conv21.bn.weight'), params.get(f'conv21.bn.bias'))
        z = z.view(z.shape[0], -1)

        z = F.linear(z, params['logits1.weight'], params['logits1.bias'])
        z = F.linear(z, params['logits2.weight'], params['logits2.bias'])

        return z





def maml_lstm_res18_fusion_train(model, support_images, support_sequences, support_labels,
                      query_images, query_sequences, query_labels, inner_step, args, optimizer, is_train=True):
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

        for _ in range(inner_step):
            # 更新权重
            support_logit = model.functional_forward(support_image, support_sequence, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - args.inner_lr * grads)
                                                   for ((name, param), grads) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        query_logit = model.functional_forward(query_image, query_sequence, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())

    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc
