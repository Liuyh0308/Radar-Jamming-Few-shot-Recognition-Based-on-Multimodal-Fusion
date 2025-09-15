import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np



class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3 , padding=1)  ## 3x3卷积  尺寸不变
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化 二分之一降采样

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


class SEblock(nn.Module):
    def __init__(self, channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)

        # 全连接层得到权重
        weight = self.fc(branch)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        # scale = weight * x     ## 得到（h,w,1,1)的weight
        # return scale
        return weight



def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

    return output




class SE_Classifier(nn.Module):      ## 五层cnn+se注意力机制模块

    def __init__(self, in_ch, n_way):
        super(SE_Classifier, self).__init__()
        self.n_class = n_way  # 分类数

    # 卷积 + 激活 + 池化
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEblock(channel=32),
            nn.MaxPool2d(2, 2)
            )

        self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                SEblock(channel=64),
                nn.MaxPool2d(2, 2)
            )

        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                SEblock(channel=128),
                nn.MaxPool2d(2, 2)
            )


        self.layer4 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                SEblock(channel=64),
                nn.MaxPool2d(2, 2)
            )


        self.layer5 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                SEblock(channel=32),
                nn.MaxPool2d(2, 2)
            )

        # 全连接层
        self.fc = nn.Sequential(
                nn.Linear(32, n_way)
            )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




    def functional_forward(self, x, params):

        x = ConvBlockFunction(x, params[f'layer1.0.weight'], params[f'layer1.0.bias'],
                              params.get(f'layer1.1.weight'), params.get(f'layer1.1.bias'))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer1.3.fc.0.weight'], params['layer1.3.fc.0.bias'])
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer1.3.fc.2.weight'], params['layer1.3.fc.2.bias'])
        x = x.view(-1,32,28,28)


        x = ConvBlockFunction(x, params[f'layer2.0.weight'], params[f'layer2.0.bias'],
                              params.get(f'layer2.1.weight'), params.get(f'layer2.1.bias'))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer2.3.fc.0.weight'], params['layer2.3.fc.0.bias'])
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer2.3.fc.2.weight'], params['layer2.3.fc.2.bias'])
        x = x.view(-1, 64, 14, 14)


        x = ConvBlockFunction(x, params[f'layer3.0.weight'], params[f'layer3.0.bias'],
                              params.get(f'layer3.1.weight'), params.get(f'layer3.1.bias'))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer3.3.fc.0.weight'], params['layer3.3.fc.0.bias'])
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer3.3.fc.2.weight'], params['layer3.3.fc.2.bias'])
        x = x.view(-1, 128, 7, 7)


        x = ConvBlockFunction(x, params[f'layer4.0.weight'], params[f'layer4.0.bias'],
                              params.get(f'layer4.1.weight'), params.get(f'layer4.1.bias'))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer4.3.fc.0.weight'], params['layer4.3.fc.0.bias'])
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer4.3.fc.2.weight'], params['layer4.3.fc.2.bias'])
        x = x.view(-1, 64, 3, 3)


        x = ConvBlockFunction(x, params[f'layer5.0.weight'], params[f'layer5.0.bias'],
                              params.get(f'layer5.1.weight'), params.get(f'layer5.1.bias'))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer5.3.fc.0.weight'], params['layer5.3.fc.0.bias'])
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, x.shape[1])
        x = F.linear(x, params['layer5.3.fc.2.weight'], params['layer5.3.fc.2.bias'])
        x = x.view(-1, 32, 1, 1)


        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['fc.0.weight'], params['fc.0.bias'])

        return x





def maml_senet_train(model, support_images, support_labels, query_images, query_labels, inner_step, args, optimizer, is_train=True):
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

    for support_image, support_label, query_image, query_label in zip(support_images, support_labels, query_images, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())
        for _ in range(inner_step):
            # Update weight
            support_logit = model.functional_forward(support_image, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - args.inner_lr * grads)
                                                   for ((name, param), grads) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        query_logit = model.functional_forward(query_image, fast_weights)
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
