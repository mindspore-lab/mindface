# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Network."""
import math
from functools import reduce
import numpy as np

from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor


# RetinaFace
def init_kaiming_uniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    """init_kaiming_uniform"""
    def _calculate_in_and_out(arr_shape):
        dim = len(arr_shape)
        if dim < 2:
            raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

        n_in = arr_shape[1]
        n_out = arr_shape[0]

        if dim > 2:

            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def calculate_gain(nonlinearity, a=None):
        """calculate_gain"""
        linear_fans = ['linear', 'conv1d', 'conv2d', 'conv3d',
                       'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fans or nonlinearity == 'sigmoid':
            return 1
        if nonlinearity == 'tanh':
            return 5.0 / 3
        if nonlinearity == 'relu':
            return math.sqrt(2.0)
        if nonlinearity == 'leaky_relu':
            if a is None:
                negative_slope = 0.01
            elif not isinstance(a, bool) and isinstance(a, int) or isinstance(a, float):
                negative_slope = a
            else:
                raise ValueError(f"negative_slope {a} not a valid number")
            return math.sqrt(2.0 / (1 + negative_slope ** 2))

        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

    fan_in, _ = _calculate_in_and_out(arr_shape)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weight = np.random.uniform(-bound, bound, arr_shape).astype(np.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(fan_in)
        bias = np.random.uniform(-bound_bias, bound_bias, arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return Tensor(weight), bias

class ConvBNReLU(nn.SequentialCell):
    """ConvBNReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer, leaky=0):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
            nn.LeakyReLU(alpha=leaky)
            # nn.ReLU()
        )

class ConvBN(nn.SequentialCell):
    """ConvBN"""
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
        )

class SSH(nn.Cell):
    """SSH"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1

        norm_layer = nn.BatchNorm2d
        self.conv3x3 = ConvBN(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, groups=1,
                              norm_layer=norm_layer)

        self.conv5x5_1 = ConvBNReLU(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv5x5_2 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.conv7x7_2 = ConvBNReLU(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv7x7_3 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.cat = P.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        """construct"""
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = self.cat((conv3x3, conv5x5, conv7x7))
        out = self.relu(out)

        return out

class FPN(nn.Cell):
    """FPN"""
    def __init__(self, cfg):
        super().__init__()
        out_channels = cfg['out_channel']
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        norm_layer = nn.BatchNorm2d
        self.output1 = ConvBNReLU(cfg['in_channel'] * 2, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)
        self.output2 = ConvBNReLU(cfg['in_channel'] * 4, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)
        self.output3 = ConvBNReLU(cfg['in_channel'] * 8, cfg['out_channel'], kernel_size=1, stride=1,
                                  padding=0, groups=1, norm_layer=norm_layer, leaky=leaky)

        self.merge1 = ConvBNReLU(cfg['out_channel'], cfg['out_channel'], kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)
        self.merge2 = ConvBNReLU(cfg['out_channel'], cfg['out_channel'], kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)

    def construct(self, input1, input2, input3):
        """construct"""
        output1 = self.output1(input1)
        output2 = self.output2(input2)
        output3 = self.output3(input3)

        up3 = P.ResizeNearestNeighbor([P.Shape()(output2)[2], P.Shape()(output2)[3]])(output3)
        output2 = up3 + output2
        output2 = self.merge2(output2)

        up2 = P.ResizeNearestNeighbor([P.Shape()(output1)[2], P.Shape()(output1)[3]])(output2)
        output1 = up2 + output1
        output1 = self.merge1(output1)

        return output1, output2, output3

class ClassHead(nn.Cell):
    """ClassHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors

        weight_shape = (self.num_anchors * 2, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0,
                                 has_bias=True, weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 2))

class BboxHead(nn.Cell):
    """BboxHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()

        weight_shape = (num_anchors * 4, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 4))

class LandmarkHead(nn.Cell):
    """LandmarkHead"""
    def __init__(self, inchannels=512, num_anchors=3):
        super().__init__()

        weight_shape = (num_anchors * 10, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        """construct"""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 10))

class RetinaFace(nn.Cell):
    """
    Build the retinaface model without loss function.

    Args:
        phase (String): Set the 'train' mode or 'val' mode. Default: 'train'
        backbone (Object): The backbone is used to extract features.
        cfg (Dict): The configuration file that contains parameters related to model.

    Examples:
        >>> backbone = resnet50(1001)
        >>> net = RetinaFace(phase='train', backbone=backbone, cfg=cfg)
    """
    def __init__(self, phase='train', backbone=None, cfg=None):

        super().__init__()
        self.phase = phase

        self.base = backbone

        self.fpn = FPN(cfg)

        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])

        self.classhead = self._make_class_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                      cfg['out_channel']], anchor_num=[2, 2, 2])
        self.bboxhead = self._make_bbox_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                    cfg['out_channel']], anchor_num=[2, 2, 2])
        self.landmarkhead = self._make_landmark_head(fpn_num=3, inchannels=[cfg['out_channel'],
                                                                            cfg['out_channel'],
                                                                            cfg['out_channel']],
                                                     anchor_num=[2, 2, 2])

        self.cat = P.Concat(axis=1)

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        """_make_class_head"""
        classhead = nn.CellList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels[i], anchor_num[i]))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        """_make_bbox_head"""
        bboxhead = nn.CellList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels[i], anchor_num[i]))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num):
        """_make_landmark_head"""
        landmarkhead = nn.CellList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels[i], anchor_num[i]))
        return landmarkhead

    def construct(self, inputs):
        """construct"""
        f1, f2, f3 = self.base(inputs)
        f1, f2, f3 = self.fpn(f1, f2, f3)

        # SSH
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        features = [f1, f2, f3]

        bbox = ()
        for i, feature in enumerate(features):
            bbox = bbox + (self.bboxhead[i](feature),)
        bbox_regressions = self.cat(bbox)

        cls = ()
        for i, feature in enumerate(features):
            cls = cls + (self.classhead[i](feature),)
        classifications = self.cat(cls)

        landm = ()
        for i, feature in enumerate(features):
            landm = landm + (self.landmarkhead[i](feature),)
        ldm_regressions = self.cat(landm)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, P.Softmax(-1)(classifications), ldm_regressions)

        return output

class RetinaFaceWithLossCell(nn.Cell):
    """
    Build the retinaface model with loss function.

    Args:
        network (Object): Retinaface model without loss function.
        multibox_loss (Object): The loss function used.
        config (Dict): The configuration file that contains parameters related to loss.

    Examples:
        >>> backbone = resnet50(1001)
        >>> net = RetinaFace(phase='train', backbone=backbone, cfg = cfg)
        >>> net = RetinaFaceWithLossCell(net, multibox_loss, config = cfg)
    """
    def __init__(self, network, multibox_loss, config):
        super().__init__()
        self.network = network
        self.loc_weight = config['loc_weight']
        self.class_weight = config['class_weight']
        self.landm_weight = config['landm_weight']
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t, landm_t):
        pred_loc, pre_conf, pre_landm = self.network(img)
        loss_loc, loss_conf, loss_landm = self.multibox_loss(pred_loc, loc_t, pre_conf, conf_t, pre_landm, landm_t)

        return loss_loc * self.loc_weight + loss_conf * self.class_weight + loss_landm * self.landm_weight
