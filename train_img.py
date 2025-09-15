import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from core.dataset import RJ_img_Dataset

from core.helper import get_model, get_dataset, adjust_learning_rate1, adjust_learning_rate2

##  选择模型
from net.maml import maml_train
from net.maml_senet import maml_senet_train
from net.maml_resnet import maml_resnet_train
from net.maml_concat import maml_concat_train
from net.maml_se_resnet import maml_se_res_train


from args import args, dev


if __name__ == '__main__':
    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.task_num, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)

    params = [p for p in model.parameters() if p.requires_grad]

    ## 这里才是模型真正进行调整的地方，是outer_lr的作用
    ## 这里为模型外层训练的地方，内层在每个net的框架中maml_train中
    start_lr1 = args.outer_lr
    start_lr2 = args.inner_lr
    optimizer = optim.Adam(params, start_lr1)
    best_acc = 0


   ## 定义外部列表，用于保存总评估值
    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []



    model.train()
    for epoch in range(args.epochs):

        ## 设置学习率自动衰减
        adjust_learning_rate1(optimizer, epoch, start_lr1)
        adjust_learning_rate2(epoch, start_lr2)
        print("outer_lr:{:.2E}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        print("inner_lr:{:.2E}".format(adjust_learning_rate2(epoch, start_lr2)))

        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []

        train_bar = tqdm(train_loader)

        ## 读取加载时频图图片数据集
        for support_images, support_labels, query_images, query_labels in train_bar:
            train_bar.set_description("epoch {}".format(epoch + 1))
            # 统一数据类型并转成cuda可读数据类型
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)

            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)


            """  该注意的细节
            注意一下这里的内层innerloop的学习率innerstep的设置为什么是1，每训练一个epoch要迭代k次，每次迭代就要进行[inner_step]次的权重更新与梯度下降
            若inner_step若设置为2,3,4。。。。n， 每次迭代要更新n次权重参数，经过实验，作者在4080（24G）上对本调制数据集上（Modulation）进行训练，结果显示数据过多，cuda内存会溢出
            ps.每张图片均为224*224*1 (经过图片转灰度图以后)
            """


            # ### 用普通卷积层作为backbone训练
            # loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
            #                        1, args, optimizer)


            # ## 用senet训练
            # loss, acc = maml_senet_train(model, support_images, support_labels, query_images, query_labels,
            #                        1, args, optimizer)



            ## 用se_resnet训练
            loss, acc = maml_se_res_train(model, support_images, support_labels, query_images, query_labels,
                                         1, args, optimizer)


            # #用concat融合cnn训练
            # loss, acc = maml_concat_train(model, support_images, support_labels, query_images, query_labels,
            #                               1, args, optimizer)



            ## 用mobilenet做backbone
            # loss, acc = maml_mobilenet_train(model, support_images, support_labels, query_images, query_labels,
            #                        1, args, optimizer)



            # #用resnet18作为backbone训练
            # loss, acc = maml_resnet_train(model, support_images, support_labels, query_images, query_labels,
            #                        1, args, optimizer)



            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))



        for support_images, support_labels, query_images, query_labels in val_loader:
            # 统一数据类型并转成cuda可读数据类型
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)

            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)

 #   该注意的细节
 # 此处又和support_set训练师设置的inner_step不一样了，因为只是对query_set进行验证，数据量不大，可以多进行几次的权重更新与梯度下降
 # ps.每张图片均为224*224*1 (经过图片转灰度图以后)

            # #用普通卷积层
            # loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
            #                        3, args, optimizer, is_train=False)


            # #用senet训练
            # loss, acc = maml_senet_train(model, support_images, support_labels, query_images, query_labels,
            #                        3, args, optimizer, is_train=False)


            ## 用se_resnet训练
            loss, acc = maml_se_res_train(model, support_images, support_labels, query_images, query_labels,
                                          3, args, optimizer, is_train=False)


            # ##用concat融合cnn训练
            # loss, acc = maml_concat_train(model, support_images, support_labels, query_images, query_labels,
            #                        3, args, optimizer, is_train=False)


            ## 用mobilenet做backbone
            # loss, acc = maml_mobilenet_train(model, support_images, support_labels, query_images, query_labels,
            #                                  3, args, optimizer, is_train=False)


            # #用resnet18
            # loss, acc = maml_resnet_train(model, support_images, support_labels, query_images, query_labels,
            #                        3, args, optimizer, is_train=False)


            # Must use .item()  to add total loss, or will occur GPU memory leak.
            # Because dynamic graph is created during forward, collect in backward.
            val_loss.append(loss.item())
            val_acc.append(acc)

        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)))


        ## 计算该epoch的平均loss和accuracy，并存储到外部列表
        all_train_acc.append(np.mean(train_acc))
        all_val_acc.append(np.mean(val_acc))
        all_train_loss.append(np.mean(train_loss))
        all_val_loss.append(np.mean(val_loss))


        if np.mean(val_acc) >= 0.98:
            # torch.save(model, './model_path/best_img_base.pth')
            # torch.save(model, './model_path/best_img_resnet.pth')
            torch.save(model, './model_path/best_img_res_se.pth')
            break


## 将每一轮结束时的acc、loss储存形成的列表打印出来进行画图
    print(all_train_acc, end='\n')
    print(all_val_acc, end='\n')
    print(all_train_loss, end='\n')
    print(all_val_loss, end='\n')




    # 保存模型路径
    # if np.mean(val_acc) > best_acc:
    #     best_acc = np.mean(val_acc)
    #     torch.save(model, 'best.pt')

    # torch.save(model, './model_path/best_img_resnet.pth')
    # torch.save(model, './model_path/best_img_res_se.pth')
    # torch.save(model, './model_path/best_img_base.pth')




