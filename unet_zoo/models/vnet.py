import torch
import torch.nn as nn
import torch.nn.functional as F

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels_initial=16, elu=True):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels_initial, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(out_channels_initial)
        self.relu1 = ELUCons(elu, out_channels_initial)
        self.out_channels_initial = out_channels_initial

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        

        if x.shape[1] == 1 and self.out_channels_initial > 1:
            x_repeated = x.repeat(1, self.out_channels_initial, 1, 1)
        elif x.shape[1] != self.out_channels_initial:

            conv_adapt = nn.Conv2d(x.shape[1], self.out_channels_initial, kernel_size=1).to(x.device)
            x_repeated = conv_adapt(x)
        else:
            x_repeated = x

        out = self.relu1(torch.add(out, x_repeated))
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm2d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        

        diffY = skipxdo.size()[2] - out.size()[2]
        diffX = skipxdo.size()[3] - out.size()[3]
        out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, num_classes, elu=True):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, num_classes, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(num_classes)
        self.relu1 = ELUCons(elu, num_classes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class VNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, elu: bool = True, nll: bool = False):
        super(VNet, self).__init__()

        self.in_tr = InputTransition(in_channels, out_channels_initial=16, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        
        self.out_tr = OutputTransition(32, num_classes, elu=elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        
        out = self.out_tr(out)
        return out
