'''
Backbone: 本部分负责引入和构建常用的骨干网络，用于后续任务
Head: 本部分为不同的任务构建不同的Head在骨干网络的基础上实现特定功能
'''

from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal, HeNormal
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore.ops import functional as F

# iresnet
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     pad_mode='pad',
                     group=groups,
                     has_bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     has_bias=False)


class IBasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Cell):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(keep_prob=1.0-dropout)
        self.fc = nn.Dense(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

        self.features.gamma = initializer(1.0, self.features.gamma.shape)
        self.features.gamma.requires_grad = False

        for m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight = initializer(Normal(0.1, 0), m.weight.shape)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.gamma = initializer(1.0, m.gamma.shape)
                m.beta = initializer(0, m.beta.shape)

        if zero_init_residual:
            for m in self.cells_and_names():
                if isinstance(m, IBasicBlock):
                    m.bn2.weight = initializer(0, m.bn2.weight.shape)

        self.reshape = ops.Reshape()
        self.flatten = ops.Flatten()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05)
            ])
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)

        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


##------------------mobilenet------------------

class Flatten(nn.Cell):
    def construct(self, x):
        return x.view(x.shape[0], -1)


class ConvBlock(nn.Cell):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.SequentialCell(
            nn.Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad', padding=padding, has_bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.PReLU(channel=out_c)
        )

    def construct(self, x):
        return self.layers(x)


class LinearBlock(nn.Cell):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0), group=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.SequentialCell(
            nn.Conv2d(in_c, out_c, kernel, group=group, stride=stride, pad_mode='pad', padding=padding, has_bias=False),
            nn.BatchNorm2d(num_features=out_c)
        )

    def construct(self, x):
        return self.layers(x)


class DepthWise(nn.Cell):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), group=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.SequentialCell(
            ConvBlock(in_c, out_c=group, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1)),
            ConvBlock(group, group, group=group, kernel=kernel, padding=padding, stride=stride),
            LinearBlock(group, out_c, kernel=(1, 1), padding=(0, 0, 0, 0), stride=(1, 1))
        )

    def construct(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Cell):
    def __init__(self, c, num_block, group, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)):
        super(Residual, self).__init__()
        Cells = []
        for _ in range(num_block):
            Cells.append(DepthWise(c, c, True, kernel, stride, padding, group))
        self.layers = nn.SequentialCell(*Cells)

    def construct(self, x):
        return self.layers(x)


class GDC(nn.Cell):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.SequentialCell(
            LinearBlock(512, 512, kernel=(7, 7), stride=(1, 1), padding=(0, 0, 0, 0), group=512),
            Flatten(),
            nn.Dense(512, embedding_size, has_bias=False),
            nn.BatchNorm1d(embedding_size))

    def construct(self, x):
        return self.layers(x)


class MobileFaceNet(nn.Cell):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.CellList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1), group=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], group=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)),
            )
        
        self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), group=128),
            Residual(64 * self.scale, num_block=blocks[1], group=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)),
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), group=256),
            Residual(128 * self.scale, num_block=blocks[2], group=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)),
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1), group=512),
            Residual(128 * self.scale, num_block=blocks[3], group=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1, 1, 1)),
        ])

        self.conv_sep = ConvBlock(128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0, 0, 0))
        self.features = GDC(num_features)
        # self.features = LinearBlock(512, 512, group=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0, 0, 0))
        # self.features = Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0, 0, 0), group=512, has_bias=False)
        self._initialize_weights()


    def _initialize_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape, cell.bias.data.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer('ones', cell.gamma.data.shape))
                cell.beta.set_data(initializer('zeros', cell.beta.data.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.data.shape, cell.weight.data.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.data.shape, cell.bias.data.dtype))


    def construct(self, x):
        for func in self.layers:
            x = func(x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)

def get_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)

##--------------------CLS Head--------------------

class PartialFC(nn.Cell):
    def __init__(self, num_classes, world_size):
        super(PartialFC, self).__init__()
        self.L2Norm = ops.L2Normalize(axis=1)
        self.weight = Parameter(initializer(
            "normal", (num_classes, 512)), name="mp_weight")
        self.sub_weight = self.weight
        self.linear = ops.MatMul(transpose_b=True).shard(
            ((1, 1), (world_size, 1)))

    def construct(self, features):
        norm_weight = self.prepare()
        total_features = self.L2Norm(features)
        logits = self.forward(total_features, norm_weight)

        return logits

    def forward(self, total_features, norm_weight):
        total_features = F.cast(total_features, mstype.float16)
        norm_weight = F.cast(norm_weight, mstype.float16)

        logits = self.linear(total_features, norm_weight)

        return F.cast(logits, mstype.float32)

    def prepare(self):
        norm_weight = self.L2Norm(self.sub_weight)
        return norm_weight


if __name__ == "__main__":
    import mindspore.numpy as np
    import mindspore as ms

    from mindspore.parallel import _cost_model_context as cost_model_context
    from mindspore import context, Tensor

    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target='GPU', save_graphs=False)
    # net = get_mbf(False, 512)
    # print(net)
    # x = ms.Tensor(np.ones([4, 3, 112, 112]), ms.float32)
    # print(x.shape)
    # output = net(x)
    # print(output.shape)

    net = iresnet100()
    print(net)
    x = ms.Tensor(np.ones([4, 3, 112, 112]), ms.float32)
    print(x.shape)
    output = net(x)
    print(output.shape)