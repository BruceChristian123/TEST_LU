#! python3.7
# -*- coding:utf-8 -*-
__author__ = 'bruce'
__time__ = '2019-08-01'
import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom的基本操作，但是扔可以通过self.vis.function
    调用原生的visdom接口
    """
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横坐标
        # 保存(loss,23)即loss的第23个点
        self.index = dict()
        self.log_text = ''

    def re_init(self, env='default', **kwargs):
        """
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        :param d: dict (name, value) i.e ('loss',0.11)
        :return:
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img_many(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        :param name:
        :param y:
        :param kwargs:
        :return:
        """
        x = self.index.get(name, 0)
        status = None if x == 0 else 'append'
        print("x: ", x, "status: ", status)
        Y =np.array([y])
        X = np.array([x])
        if name.lower() == 'loss':
            self.vis.line(Y=Y, X=X, win=name, opts=dict(title=name, xlabel='batch', ylabel='loss'), update=status, **kwargs)
        elif name.lower() == 'val_accuracy':
            self.vis.line(Y=Y, X=X, win=name, opts=dict(title=name, xlabel='epoch', ylabel='val accuracy'), update=status, **kwargs)
        else:
            self.vis.line(Y=Y, X=X, win=name, opts=dict(title=name), update=status, **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """

        :param name:
        :param img_:
        :param kwargs:
        :return:
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss': 1, 'lr':0.0001})
        :param info:
        :param win:
        :return:
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
