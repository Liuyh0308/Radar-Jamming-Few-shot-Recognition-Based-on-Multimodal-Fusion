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


def ConvBlockFunction(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

    return output

def ConvBlockFunction2(input, w, b, w_bn, b_bn):   #in_ch在help.py中设置
    x = F.conv2d(input, w, b, padding=1)
    x = torch.unsqueeze(x, 0)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

    return output



class Concat_Classifier(nn.Module):
    def __init__(self, in_ch, n_way):
        super(Concat_Classifier, self).__init__()




    def forward(self, x):

        a4 = a3.view(a3.shape[0], -1)


        b = self.logits(a4)

        return b





    def functional_forward(self, x, params):



        b = F.linear(a4, params['logits.weight'], params['logits.bias'])

        return b




def maml_concat_train(model, support_images, support_labels, query_images, query_labels, inner_step, args, optimizer, is_train=True):
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
