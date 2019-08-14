#! python3.6
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-07-31'
import os
import platform
from pathlib import Path
import _pickle as cpickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from visualizer import Visualizer
import Model
from config import DefaultConfig
from Dataset.lung_dataset import LungNodule


def write_csv(result, fname):
    """
    将result写入到文件fname中
    :param result: DataFrame类型
    :param fname: 文件名，包括具体路径
    :return:
    """
    file_path = os.path.join(os.getcwd(), fname)
    result.to_csv(file_path, sep=',', index=False)
    return


@torch.no_grad()
def val(opt, model,data_loader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in enumerate(data_loader):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def train(opt, **kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)
    # 1. 选择model
    model = getattr(Model, opt.model)().eval()
    pth_name = None
    if opt.load_model_path:
        fname = Path.cwd() / 'checkpoints' / opt.load_model_path
        if fname.is_file() and Path(fname).exists():
            pth_name = str(fname)
            model.load(pth_name)
    model.to(opt.device)

    # 2. data
    train_data = LungNodule(opt.train_data_root, train=True)  # 训练集
    val_data = LungNodule(opt.train_data_root, train=False, test=False)  # 验证集
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 3. criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.learning_rate
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # 4. meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)  # 混淆矩阵
    previous_loss = 1e10
    epoch_list = []
    ii_list = []
    loss_list = []
    # 5. train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_loader):

            # train model
            data = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, target)
            print("epoch:{epoch}, ii:{ii}, loss:{loss_item}".format(epoch=epoch, ii=ii, loss_item=loss.item()))
            epoch_list.append(epoch)
            ii_list.append(ii)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii+1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save(pth_name)

        # validate and visualize
        val_cm, val_accuracy = val(opt, model, val_loader)
        print("val_accuracy:", val_accuracy)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]
    vis_name = Path.cwd() / 'checkpoints' / 'train_vis.dat'
    with open(str(vis_name), 'wb') as f:
        cpickle.dump(vis, f)
    f.close()
    loss_dict = {'epoch': epoch_list, 'ii': ii_list, 'loss': loss_list}
    df = pd.DataFrame(loss_dict)
    loss_name = Path.cwd() / 'checkpoints' / 'loss_log.csv'
    df.to_csv(str(loss_name), sep=',')
    return loss_meter, confusion_matrix, previous_loss


@torch.no_grad()
def test(opt, **kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(Model, opt.model)().eval()
    if opt.load_model_path:
        fname = Path.cwd() / 'checkpoints' / opt.load_model_path
        if fname.is_file() and Path(fname).exists():
            pth_name = str(fname)
            model.load(pth_name)
    model.to(opt.device)

    # data
    test_data = LungNodule(opt.test_data_root, train=False, test=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    results = []
    for ii, (data, y) in enumerate(test_loader):
        x = data.to(opt.device)
        score = model(x)
        probability = torch.nn.functional.softmax(score, dim=1)
        label = probability.max(dim=1)[1].tolist()
        batch_results = [(y_.item(), label_) for y_, label_ in zip(y, label)]
        # batch_results = [(y_.item(), probability_) for y_, probability_ in zip(y, probability)]
        results.extend(batch_results)
    df = pd.DataFrame(results, columns=['y_true', 'y_pred'])
    write_csv(df, opt.result_file)
    return df


if __name__ == '__main__':
    if platform.system() == 'Windows':
        data_path = 'E:/Luna16/dataset'
    elif platform.system() == 'Darwin':
        data_path = '/Users/bruce/OneDrive/dataset'
    else:
        data_path = '/dataset'
    param = {'train_data_root': data_path, 'test_data_root': data_path}
    opt = DefaultConfig()
    loss_meter, confusion_matrix, previous_loss = train(opt, **param)
    df = test(opt, **param)
