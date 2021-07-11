import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d( in_channels = 1, out_channels = 1, kernel_size = (1, 4) )
        self.conv2 = nn.Conv2d( in_channels = 1, out_channels = 1, kernel_size = (1, 4) )
        self.conv3 = nn.Conv2d( in_channels = 1, out_channels = 1, kernel_size = (1, 4) )

    def forward(self, states):
        out1 = self.conv1(states)
        out2 = self.conv2(out1)
        out = self.conv3(out2)

        return out

##### Generator #####

class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dimension, use_dropout=False, use_bias=False):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dimension, use_dropout, use_bias)

    def build_conv_block(self, dimension, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dimension, dimension, kernel_size=3, padding=1, bias=use_bias),
                       nn.BatchNorm2d(dimension),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(dimension, dimension, kernel_size=3, padding=1, bias=use_bias),
                       nn.BatchNorm2d(dimension)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.inc = Inconv(input_nc, ngf)
        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 4)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, use_dropout=use_dropout)]
        self.resblocks = nn.Sequential(*model)

        self.up1 = Up(ngf * 4, ngf * 2)
        self.up2 = Up(ngf * 2, ngf)

        self.outc = Outconv(ngf, output_nc)

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        out['d2'] = self.down2(out['d1'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])

        return self.outc(out['u2'])

##### Discriminator #####

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, use_sigmoid=False, use_bias=False):
        super(PixelDiscriminator, self).__init__()

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        self.sequence = [ nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True) ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            self.sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*self.sequence)

    def forward(self, input):
        return self.model(input)