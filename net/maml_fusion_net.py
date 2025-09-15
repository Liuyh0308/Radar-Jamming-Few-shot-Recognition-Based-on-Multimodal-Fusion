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


def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

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


class Image_Brunch(nn.Module):

    def __init__(self, in_ch):
        super(Image_Brunch, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)  ## 如果要用4层cnn+1层linear imgsize=（28,28）
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.conv5 = ConvBlock(64, 64)
        self.conv6 = ConvBlock(64, 64)
        self.conv7 = ConvBlock(64, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x

class Sequence_Brunch(nn.Module):

    def __init__(self, in_ch):
        super(Sequence_Brunch, self).__init__()
        self.conv8 = ConvBlock1(in_ch, 16)
        self.conv9 = ConvBlock1(16, 32)
        self.conv10 = ConvBlock1(32, 64)
        self.conv11 = ConvBlock1(64, 64)
        self.conv12 = ConvBlock1(64, 64)
        self.conv13 = ConvBlock1(64, 64)
        self.conv14 = ConvBlock1(64, 64)
        self.conv15 = ConvBlock1(64, 64)
        self.conv16 = ConvBlock1(64, 64)
        self.conv17 = ConvBlock1(64, 64)

    def forward(self, y):
        y = self.conv8(y)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.conv11(y)
        y = self.conv12(y)
        y = self.conv13(y)
        y = self.conv14(y)
        y = self.conv15(y)
        y = self.conv16(y)
        y = self.conv17(y)

        return y



class Fusion_Classifier(nn.Module):          ## baseline 四层简单卷积层 可在通道数上做改动 可不变 也可先升维再降维
    def __init__(self, in_ch1, in_ch2, n_way):
        super(Fusion_Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch1, 8)       ## 如果要用4层cnn+1层linear imgsize=（28,28）
        self.conv2 = ConvBlock(8, 16)
        self.conv3 = ConvBlock(16, 32)
        self.conv4 = ConvBlock(32, 32)
        self.conv5 = ConvBlock(32, 16)
        self.conv6 = ConvBlock(16, 8)
        self.conv7 = ConvBlock(8, 8)

        self.conv8 = ConvBlock1(in_ch2, 8)
        self.conv9 = ConvBlock1(8, 16)
        self.conv10 = ConvBlock1(16, 32)
        self.conv11 = ConvBlock1(32, 64)
        self.conv12 = ConvBlock1(64, 64)
        self.conv13 = ConvBlock1(64, 64)
        self.conv14 = ConvBlock1(64, 32)
        self.conv15 = ConvBlock1(32, 16)
        self.conv16 = ConvBlock1(16, 8)
        self.conv17= ConvBlock1(8, 8)

        # self.lstm1 = nn.LSTM()
        self.conv18 = ConvBlock2(16, 16)
        self.conv19 = ConvBlock2(16, 16)

        self.logits = nn.Linear(16, n_way)


    def forward(self, x, y):
        ## 图片输入卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)

        ## 序列输入卷积
        y = self.conv8(y)
        y = self.conv9(y)
        y = self.conv10(y)
        y = self.conv11(y)
        y = self.conv12(y)
        y = self.conv13(y)
        y = self.conv14(y)
        y = self.conv15(y)
        y = self.conv16(y)
        # y = self.conv17(y)

        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        z = torch.concat(x, y)
        z = self.conv18(z)
        z = self.conv19(z)

        z = self.logits(z)

        return z


    def functional_forward(self, x, y, params):
        x = ConvBlockFunction(x, params[f'conv1.conv2d.weight'], params[f'conv1.conv2d.bias'],
                              params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv2.conv2d.weight'], params[f'conv2.conv2d.bias'],
                              params.get(f'conv2.bn.weight'), params.get(f'conv2.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv3.conv2d.weight'], params[f'conv3.conv2d.bias'],
                              params.get(f'conv3.bn.weight'), params.get(f'conv3.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv4.conv2d.weight'], params[f'conv4.conv2d.bias'],
                              params.get(f'conv4.bn.weight'), params.get(f'conv4.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv5.conv2d.weight'], params[f'conv5.conv2d.bias'],
                              params.get(f'conv5.bn.weight'), params.get(f'conv5.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv6.conv2d.weight'], params[f'conv6.conv2d.bias'],
                              params.get(f'conv6.bn.weight'), params.get(f'conv6.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv7.conv2d.weight'], params[f'conv7.conv2d.bias'],
                              params.get(f'conv7.bn.weight'), params.get(f'conv7.bn.bias'))


        y = ConvBlockFunction1(y, params[f'conv8.conv1d.weight'], params[f'conv8.conv1d.bias'],
                              params.get(f'conv8.bn.weight'), params.get(f'conv8.bn.bias'))
        y = ConvBlockFunction1(y, params[f'conv9.conv1d.weight'], params[f'conv9.conv1d.bias'],
                              params.get(f'conv9.bn.weight'), params.get(f'conv9.bn.bias'))
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

        x = x.view(x.shape[0], x.shape[1], -1)
        # y = y.view(y.shape[0], -1)

        z = torch.concat((x, y), dim=1)
        z = ConvBlockFunction2(z, params[f'conv18.conv1d.weight'], params[f'conv18.conv1d.bias'],
                               params.get(f'conv18.bn.weight'), params.get(f'conv18.bn.bias'))
        z = ConvBlockFunction2(z, params[f'conv19.conv1d.weight'], params[f'conv19.conv1d.bias'],
                               params.get(f'conv19.bn.weight'), params.get(f'conv19.bn.bias'))
        z = z.view(z.shape[0], -1)

        z = F.linear(z, params['logits.weight'], params['logits.bias'])

        return z





def maml_fusion_train(model, support_images, support_sequences, support_labels,
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
