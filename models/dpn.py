import chainer
import chainer.functions as F
import chainer.links as L

from collections import OrderedDict
from .sequential import Sequential


class GroupedConvolution2D(chainer.ChainList):
    def __init__(self, in_chs, out_chs, ksize=None, stride=1, pad=0, groups=1, nobias=False, initialW=None, initial_bias=None):
        assert in_chs % groups == 0
        assert out_chs % groups == 0
        group_in_chs = int(in_chs / groups)
        group_out_chs = int(in_chs / groups)
        super(GroupedConvolution2D, self).__init__(
            *[L.Convolution2D(group_in_chs, group_out_chs, ksize, stride, pad, nobias, initialW, initial_bias) for _ in range(groups)]
        )
        self.group_in_chs = group_in_chs

    def __call__(self, x):
        return F.concat([f(x[:,i*self.group_in_chs:(i+1)*self.group_in_chs,:,:]) for i, f in enumerate(self.children())], axis=1)


class DualPathBlock(chainer.Chain):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if _type is 'proj':
            key_stride = 1
            self.has_proj = True
        if _type is 'down':
            key_stride = 2
            self.has_proj = True
        if _type is 'normal':
            key_stride = 1
            self.has_proj = False

        with self.init_scope():
            if self.has_proj:
                self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*inc, kernel_size=1, stride=key_stride)

            self.layers = Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=G)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+inc, kernel_size=1, stride=1)),
            ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        if groups==1:
            return Sequential(OrderedDict([
                ('norm', L.BatchNormalization(in_chs)),
                ('relu', F.relu),
                ('conv', L.Convolution2D(in_chs, out_chs, kernel_size, stride, padding, nobias=True)),
            ]))
        else:
            return Sequential(OrderedDict([
                ('norm', L.BatchNormalization(in_chs)),
                ('relu', F.relu),
                ('conv', GroupedConvolution2D(in_chs, out_chs, kernel_size, stride, padding, groups, nobias=True)),
            ]))
            

    def __call__(self, x):
        data_in = F.concat(x, axis=1) if isinstance(x, list) else x
        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:,:self.num_1x1_c,:,:]
            data_o2 = data_o[:,self.num_1x1_c:,:,:]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)

        summ = data_o1 + out[:,:self.num_1x1_c,:,:]
        dense = F.concat([data_o2, out[:,self.num_1x1_c:,:,:]], axis=1)
        return [summ, dense]


class MaxPooling2D(object):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True):
        self.args = [ksize, stride, pad, cover_all]

    def __call__(self, x):
        return F.max_pooling_2d(x, *self.args)


class DPN92(chainer.Chain):

    def __init__(self, class_labels=10):
        self.k_R = 96
        self.num_init_features = 64
        self.g = 32
        self.k_sec = (3, 4, 20, 3)
        self.inc_sec = (16, 32, 24, 128)

        blocks = OrderedDict()

        blocks['conv1'] = Sequential(
            L.Convolution2D(3, self.num_init_features, ksize=7, stride=2, pad=3, nobias=True),
            L.BatchNormalization(self.num_init_features),
            F.relu,
            MaxPooling2D(ksize=3, stride=2, pad=1)
        )

        bw = 256
        inc = self.inc_sec[0]
        R = int((self.k_R*bw)/256)
        blocks['conv2_1'] = DualPathBlock(self.num_init_features, R, R, bw, inc, self.g, 'proj')
        in_chs = bw + 3 * inc
        for i in range(2, self.k_sec[0]+1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'normal')
            in_chs += inc

        bw = 512
        inc = self.inc_sec[1]
        R = int((self.k_R*bw)/256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, self.k_sec[1]+1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'normal')
            in_chs += inc

        bw = 1024
        inc = self.inc_sec[2]
        R = int((self.k_R*bw)/256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, self.k_sec[2]+1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'normal')
            in_chs += inc

        bw = 2048
        inc = self.inc_sec[3]
        R = int((self.k_R*bw)/256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, self.k_sec[3]+1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, self.g, 'normal')
            in_chs += inc

        with self.init_scope():
            self.features = Sequential(blocks)
            self.classifier = L.Linear(in_chs, class_labels)


    def __call__(self, x):
        features = F.concat(self.features(x), axis=1)
        out = F.average_pooling_2d(features, ksize=7)
        out = self.classifier(out)
        return out
