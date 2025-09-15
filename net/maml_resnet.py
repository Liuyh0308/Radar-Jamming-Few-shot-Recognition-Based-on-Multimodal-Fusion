import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):

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


    def forward(self, x):
        out = self.left(x)
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

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, stride=2, padding=3)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    return output

# def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
#     x = F.conv2d(input, w, b, stride=2, padding=3)
#     x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
#     x = F.relu(x)
#     output = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
#
#     return output


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



class Res_Classifier(nn.Module):     ##Resnet18
    def __init__(self, in_ch, n_way):
        super(Res_Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ResidualBlock(64, 64)
        self.conv3 = ResidualBlock(64, 64)
        self.conv4 = ResidualBlock1(64, 128)
        self.conv5 = ResidualBlock(128, 128)
        self.conv6 = ResidualBlock1(128, 256)
        self.conv7 = ResidualBlock(256, 256)
        self.conv8 = ResidualBlock1(256, 512)
        self.conv9 = ResidualBlock(512, 512)


        self.logits = nn.Linear(512, n_way)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = F.avg_pool2d(x, kernel_size=7)

        x = x.view(x.shape[0], -1)
        x = self.logits(x)

        return x



    def functional_forward(self, x, params):

        x = ConvBlockFunction(x, params['conv1.conv2d.weight'], params[f'conv1.conv2d.bias'],
                                  params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))


        x1 = ResidualBlockFunction(x, params[f'conv2.left.0.weight'], params[f'conv2.left.0.bias'],
                                  params.get(f'conv2.left.1.weight'), params.get(f'conv2.left.1.bias'))
        x1 = ResidualBlockFunction_1(x1, params[f'conv2.left.3.weight'], params[f'conv2.left.3.bias'],
                                   params.get(f'conv2.left.4.weight'), params.get(f'conv2.left.4.bias'))
        x1 += x
        x1 = F.relu(x1)


        x2 = ResidualBlockFunction(x1, params[f'conv3.left.0.weight'], params[f'conv3.left.0.bias'],
                                   params.get(f'conv3.left.1.weight'), params.get(f'conv3.left.1.bias'))
        x2 = ResidualBlockFunction_1(x2, params[f'conv3.left.3.weight'], params[f'conv3.left.3.bias'],
                                   params.get(f'conv3.left.4.weight'), params.get(f'conv3.left.4.bias'))
        x2 += x1
        x2= F.relu(x2)


        x3_0 = ResidualBlockFunction1_2(x2, params[f'conv4.shortcut.0.weight'], params[f'conv4.shortcut.0.bias'],
                                        params.get(f'conv4.shortcut.1.weight'), params.get(f'conv4.shortcut.1.bias'))
        x3 = ResidualBlockFunction1(x2, params[f'conv4.left.0.weight'], params[f'conv4.left.0.bias'],
                                    params.get(f'conv4.left.1.weight'), params.get(f'conv4.left.1.bias'))

        x3 = ResidualBlockFunction1_1(x3, params[f'conv4.left.3.weight'], params[f'conv4.left.3.bias'],
                                    params.get(f'conv4.left.4.weight'), params.get(f'conv4.left.4.bias'))
        x3 += x3_0
        x3 = F.relu(x3)


        x4 = ResidualBlockFunction(x3, params[f'conv5.left.0.weight'], params[f'conv5.left.0.bias'],
                                    params.get(f'conv5.left.1.weight'), params.get(f'conv5.left.1.bias'))
        x4 = ResidualBlockFunction_1(x4, params[f'conv5.left.3.weight'], params[f'conv5.left.3.bias'],
                                    params.get(f'conv5.left.4.weight'), params.get(f'conv5.left.4.bias'))
        x4 += x3
        x4 = F.relu(x4)


        x5_0 = ResidualBlockFunction1_2(x4, params[f'conv6.shortcut.0.weight'], params[f'conv6.shortcut.0.bias'],
                                        params.get(f'conv6.shortcut.1.weight'), params.get(f'conv6.shortcut.1.bias'))
        x5 = ResidualBlockFunction1(x4, params[f'conv6.left.0.weight'], params[f'conv6.left.0.bias'],
                                    params.get(f'conv6.left.1.weight'), params.get(f'conv6.left.1.bias'))

        x5 = ResidualBlockFunction1_1(x5, params[f'conv6.left.3.weight'], params[f'conv6.left.3.bias'],
                                      params.get(f'conv6.left.4.weight'), params.get(f'conv6.left.4.bias'))
        x5 += x5_0
        x5 = F.relu(x5)


        x6 = ResidualBlockFunction(x5, params[f'conv7.left.0.weight'], params[f'conv7.left.0.bias'],
                                   params.get(f'conv7.left.1.weight'), params.get(f'conv7.left.1.bias'))
        x6 = ResidualBlockFunction_1(x6, params[f'conv7.left.3.weight'], params[f'conv7.left.3.bias'],
                                     params.get(f'conv7.left.4.weight'), params.get(f'conv7.left.4.bias'))
        x6 += x5
        x6 = F.relu(x6)


        x7_0 = ResidualBlockFunction1_2(x6, params[f'conv8.shortcut.0.weight'], params[f'conv8.shortcut.0.bias'],
                                        params.get(f'conv8.shortcut.1.weight'), params.get(f'conv8.shortcut.1.bias'))
        x7 = ResidualBlockFunction1(x6, params[f'conv8.left.0.weight'], params[f'conv8.left.0.bias'],
                                    params.get(f'conv8.left.1.weight'), params.get(f'conv8.left.1.bias'))

        x7 = ResidualBlockFunction1_1(x7, params[f'conv8.left.3.weight'], params[f'conv8.left.3.bias'],
                                      params.get(f'conv8.left.4.weight'), params.get(f'conv8.left.4.bias'))
        x7 += x7_0
        x7 = F.relu(x7)


        x8 = ResidualBlockFunction(x7, params[f'conv9.left.0.weight'], params[f'conv9.left.0.bias'],
                                   params.get(f'conv9.left.1.weight'), params.get(f'conv9.left.1.bias'))
        x8 = ResidualBlockFunction_1(x8, params[f'conv9.left.3.weight'], params[f'conv9.left.3.bias'],
                                     params.get(f'conv9.left.4.weight'), params.get(f'conv9.left.4.bias'))
        x8 += x7
        x8 = F.relu(x8)


        x9 = F.avg_pool2d(x8, kernel_size=7)
        x9 = x9.view(x9.size(0), -1)
        x9 = F.linear(x9, params['logits.weight'], params['logits.bias'])

        return x9





def maml_resnet_train(model, support_images, support_labels, query_images, query_labels, inner_step, args, optimizer, is_train=True):
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


