import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt


def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.conv_out1 = nn.Conv2d(64, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        #self.conv_out2 = nn.Conv2d(64, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.conv_out3 = nn.Conv2d(64, 3 * upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        y = self.relu(self.conv2(x))
        x = x + self.relu(self.conv3(y))
        y = self.relu(self.conv4(x))
        x = x + self.relu(self.conv5(y))
        #out1 = self.relu(self.conv_out1(x))
        y = self.relu(self.conv6(x))
        x = x + self.relu(self.conv7(y))
        #out2 = self.relu(self.conv_out2(x))
        y = self.relu(self.conv8(x))
        x = x + self.relu(self.conv9(y))
        out3 = self.relu(self.conv_out3(x))

        #ut1 = self.pixel_shuffle(out1)
        #ut2 = self.pixel_shuffle(out2)
        out3 = self.pixel_shuffle(out3)

        return out3

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight) * sqrt(2))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight) * sqrt(2))
        self.conv3.weight.data.copy_(_get_orthogonal_init_weights(self.conv3.weight) * sqrt(2))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight) * sqrt(2))
        self.conv5.weight.data.copy_(_get_orthogonal_init_weights(self.conv5.weight) * sqrt(2))
        self.conv6.weight.data.copy_(_get_orthogonal_init_weights(self.conv6.weight) * sqrt(2))
        self.conv7.weight.data.copy_(_get_orthogonal_init_weights(self.conv6.weight) * sqrt(2))
        self.conv8.weight.data.copy_(_get_orthogonal_init_weights(self.conv6.weight) * sqrt(2))
        self.conv9.weight.data.copy_(_get_orthogonal_init_weights(self.conv6.weight) * sqrt(2))
        #self.conv_out1.weight.data.copy_(_get_orthogonal_init_weights(self.conv_out1.weight))
       # self.conv_out2.weight.data.copy_(_get_orthogonal_init_weights(self.conv_out2.weight))
        self.conv_out3.weight.data.copy_(_get_orthogonal_init_weights(self.conv_out3.weight))
