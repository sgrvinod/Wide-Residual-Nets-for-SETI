import math
from torch import nn


class BasicBlock(nn.Module):
    """
    Graph of the Basic Block, as defined in the paper.

    This block contains two 3x3 convolutional layers, each with prior Batch Norm and ReLU.

    There is an additive residual connection across the block.
    If the number of dimensions change across the block, this residual is a convolutional projection of the input.

    Args:
        inplanes (int): number of dimensions in the input tensor.
        outplanes (int): number of dimensions in the output tensor.
        stride (int): stride length for the filter.
        dropout (float, fraction): the fraction of neurons to randomly drop/set to zero in-between conv. layers.
    """

    def __init__(self, inplanes, outplanes, stride, dropout=0.0):
        super(BasicBlock, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout
        if self.inplanes != self.outplanes:
            self.projection = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.projection = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        if self.inplanes != self.outplanes:
            residual = self.projection(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout > 0.:
            out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        out += residual
        return out


class BottleNeck(nn.Module):
    """
    Graph of the Bottleneck Block, as defined in the paper.

    This block contains (in order) 1x1, 3x3, 1x1 convolutional layers, each with prior Batch Norm and ReLU.
    The first 1x1 reduces the number of dimensions, before the second 1x1 increases it to the expected output dimensions.

    There is an additive residual connection across the block.
    If the number of dimensions change across the block, this residual is a convolutional projection of the input.

    Args:
        inplanes (int): number of dimensions in the input tensor.
        outplanes (int): number of dimensions in the output tensor.
        stride (int): stride length for the filter.
        dropout (float, fraction): the fraction of neurons to randomly drop/set to zero in-between conv. layers.
    """

    def __init__(self, inplanes, outplanes, stride, dropout=0.0):
        super(BottleNeck, self).__init__()

        self.inplanes = inplanes
        self.outplanes = outplanes

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes / 4, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes / 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes / 4, outplanes / 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes / 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outplanes / 4, outplanes, kernel_size=1, stride=1, padding=1, bias=False)
        self.dropout = dropout
        if self.inplanes != self.outplanes:
            self.projection = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.projection = None

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        if self.inplanes != self.outplanes:
            residual = self.projection(out)
        else:
            residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.dropout > 0.:
            out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        if self.dropout > 0.:
            out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = self.conv3(out)
        out += residual
        return out


class WideResNet(nn.Module):
    """
    Graph of the Wide Residual Network, as defined in the paper.

    This network contains 4 convolutional blocks, each increasing dimensions by a factor of 'k':
    The first is a single 3x3 Convolution, increasing dimensions from 2 (log(amplitude^2), phase) to 16.
    The second is a sequence of Basic or BottleNeck Blocks, 16 dimensions -> 16*k
    The third is a sequence of Basic or BottleNeck Blocks, 16*k dimensions -> 16*k^2
    The fourth is a sequence of Basic or BottleNeck Blocks, 16*k dimensions -> 16*k^3

    These convolutional layers are followed by Batch Norm, ReLU, Average Pool, and finally a Fully Connected Layer
    to perform the classification.

    Args:
        n (int): number of single convolutional layers in the entire network, 'n' in the paper.
        k (int): widening factor for each succeeding convolutional layer, 'k' in the paper.
        block (nn.module): BasicBlock or BottleneckBlock.
        dropout (float, fraction): the fraction of neurons to randomly drop/set to zero inside the blocks.
    """

    def __init__(self, n, k, block=BasicBlock, dropout=0.0):
        super(WideResNet, self).__init__()

        if (n - 4) % 6 != 0:
            raise ValueError("Invalid depth! Depth must be (6 * n_blocks + 4).")
        n_blocks = (n - 4) / 6

        self.conv_block1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_block2 = self._make_layer(block, n_blocks, 16, 16 * k, 2, dropout)
        self.conv_block3 = self._make_layer(block, n_blocks, 16 * k, 32 * k, 2, dropout)
        self.conv_block4 = self._make_layer(block, n_blocks, 32 * k, 64 * k, 2, dropout)
        self.bn1 = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * k * 6 * 8, 7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_weights = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n_weights))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, n_blocks, inplanes, outplanes, stride, dropout):
        """
        Graph of a Convolutional block layer (conv_block2/conv_block3/conv_block4), as defined in the paper.

        This graph assembles a number of blocks (BasicBlock or BottleNeck) in sequence.

        Args:
            block (nn.module): BasicBlock or ResidualBlock.
            inplanes (int): number of dimensions in the input tensor.
            outplanes (int): number of dimensions in the output tensor.
            stride (int): stride length for the filter.
            dropout (float, fraction): the fraction of neurons to randomly drop/set to zero in-between conv. layers.
            """
        layers = []
        for i in range(n_blocks):
            if i == 0:
                layers.append(block(inplanes, outplanes, stride, dropout))
            else:
                layers.append(block(outplanes, outplanes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def wresnet34x2():
    model = WideResNet(n=34, k=2, block=BasicBlock, dropout=0.3)
    return model
