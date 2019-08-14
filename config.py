#! python3.6
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-07-18'
import platform
import warnings
import torch
if platform.system() == 'Darwin':
    src_path = '/Volumes/本地磁盘/论文数据库/LUNA16'
    csv_name = '/Volumes/本地磁盘/论文数据库/LUNA16/CSVFILES/annotations.csv'
    dist_path = '/Volumes/LUNA16/mask'
    process_path = '/Volumes/LUNA16/process'
    segmentation_path = '/Volumes/本地磁盘/论文数据库/LUNA16-PREPARE'
    subset_count = 1
    candidates_csv = '/Volumes/本地磁盘/论文数据库/LUNA16/CSVFILES/candidates.csv'
    cube_size = 48
elif platform.system() == 'Windows':
    src_path = 'E:/Luna16'
    csv_name = 'E:/Luna16/CSVFILES/annotations.csv'
    dist_path = 'E:/Luna16/mask'
    segmentation_path = 'D:/LUNA16-PREPARE'
    subset_count = 1
    candidates_csv = 'E:/Luna16/CSVFILES/candidates.csv'
    cube_size = 48


class DefaultConfig(object):
    env = 'default'  # visdom环境
    vis_port = 8097  # visdom端口
    model = 'DenseNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/dataset'  # 训练集的存放路径
    test_data_root = '/dataset'  # 测试集的存放路径
    load_model_path = 'DenseNet_0814_14_21_00.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 8  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # is os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    learning_rate = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr * lr_decay
    weight_decay = 1e-4  # 损失函数

    def _parse(self, kwargs):
        """
        根据字典kwargs更新config参数
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            # if not hasattr(self, k):
            #     warnings.warn("Warning: opt has not attribute {k}")
            setattr(self, k, v)

        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
        #     if not k.startswith('_'):
        #         print(k. getattr(self, k))

# opt = DefaultConfig()
