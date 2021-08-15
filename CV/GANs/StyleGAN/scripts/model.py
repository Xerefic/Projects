import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)

        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, x):
        return self.conv(x)

class EqualizedLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EqualizedLinear, self).__init__()
        linear = nn.Linear(in_channels, out_channels)
        
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)

class AdaIn(nn.Module):
    def __init__(self, style_dim, channel):
        super(AdaIn, self).__init__()

        self.channel = channel

        self.instance_norm = nn.InstanceNorm2d(channel)
        self.linear = EqualizedLinear(style_dim, channel * 2)

    def forward(self, x, style):
        style = self.linear(style).view(2, -1, self.channel, 1, 1)

        x = self.instance_norm(x)
        x = (x * (style[0] + 1)) + style[1]

        return x

class NoiseInjection_Util(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection_Util, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x, noise):
        return x + self.weight * noise

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()

        injection = NoiseInjection_Util(channel)
        self.injection = equal_lr(injection)

    def forward(self, x, noise):
        return self.injection(x, noise)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class minibatch_stddev_layer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super(minibatch_stddev_layer, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        group_size = min(self.group_size, x.size(0))
        origin_shape = x.shape
        # split group
        y = x.view(
            group_size,
            -1, 
            self.num_new_features, 
            origin_shape[1] // self.num_new_features, 
            origin_shape[2], 
            origin_shape[3]
        )
        
        # calculate stddev over group
        y = torch.sqrt(torch.mean((y - torch.mean(y, dim=0, keepdim=True)) ** 2, dim=0) + 1e-8)
        # [G, F. C, H, W]
        y = torch.mean(y, dim=[2,3,4], keepdim=True)
        # [G, F, 1, 1, 1]
        y = torch.squeeze(y, dim=2)
        # [G, F, 1, 1] 
        y = y.repeat(group_size, 1, origin_shape[2], origin_shape[3])
        # [B, F, H, W]
        
        return torch.cat([x, y], dim=1)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, prev=None):
        super(UpBlock, self).__init__()

        self.prev = prev

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if prev:
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)
        else:
            self.input = nn.Parameter(torch.randn(1, out_channel, 4, 4))

        self.noisein1 = NoiseInjection(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.adain1 = AdaIn(style_dim, out_channel)

        self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)
        self.noisein2 = NoiseInjection(out_channel)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaIn(style_dim, out_channel)

        self.to_rgb = EqualizedConv2d(out_channel, 3, 1, 1, 0)

    # if last layer (0 <= alpha <= 1) -> return RGB image (3 channels)
    # else return feature map of prev layer
    def forward(self, x, style, alpha=-1.0, noise=None):
        if self.prev: # if module has prev, then forward first.
            w, style = style[-1], style[:-1] # pop last style
            prev_x = x = self.prev(x, style)

            x = self.upsample(x)

            x = self.conv1(x)
        else: # else initial constant
            w = style[0]
            x = self.input.repeat(w.size(0), 1, 1, 1)
            
        noise = noise if noise else torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        x = self.noisein1(x, noise) 
        x = self.lrelu1(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        x = self.noisein2(x, noise) 
        x = self.lrelu2(x)
        x = self.adain2(x, w)

        if 0.0 <= alpha < 1.0:
            prev_rgb = self.prev.to_rgb(self.upsample(prev_x))
            x = alpha * self.to_rgb(x) + (1 - alpha) * prev_rgb
        elif alpha == 1:
            x = self.to_rgb(x)

        return x

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, next=None):
        super(DownBlock, self).__init__()

        self.next = next

        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

        if next:
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)
        else:
            self.conv1 = nn.Sequential(
                minibatch_stddev_layer(),
                EqualizedConv2d(in_channel + 1, out_channel, 3, 1, 1),
            )
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 4, 1, 0)

            self.linear = EqualizedLinear(out_channel, 1)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.from_rgb = EqualizedConv2d(3, in_channel, 1, 1, 0)

    def forward(self, x, alpha=-1.0):
        input = x

        if 0 <= alpha:
            x = self.from_rgb(x)

        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)

        if self.next:
            x = self.downsample(x)

            if 0.0 <= alpha < 1.0:
                input = self.downsample(input)
                x = alpha * x + (1 - alpha) * self.next.from_rgb(input)

            x = self.next(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        
        return x

##### Generator #####

class Generator(nn.Module):
    def __init__(self, channels, style_dim, style_depth):
        super(Generator, self).__init__()
        
        self.style_dim = style_dim
        self.now_growth = 1
        self.channels = channels

        self.model = UpBlock(channels[0], channels[1], style_dim, prev=None)

        layers = [PixelNorm()]
        for _ in range(style_depth):
            layers.append(EqualizedLinear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style_mapper = nn.Sequential(*layers)

    def forward(self, z, alpha):
        if type(z) not in (tuple, list):
            w = self.style_mapper(z)
            w = [w for _ in range(self.now_growth)]
        else:
            assert len(z) == 2  # now, only support mix two styles
            w1, w2 = self.style_mapper(z[0]), self.style_mapper(z[1])
            point = random.randint(1, self.now_growth-1)
            # layer_0 ~ layer_p: style with w1
            # layer_p ~ layer_n: style with w2
            w = [w1 for _ in range(point)] + [w2 for _ in range(point, self.now_growth)]

        x = self.model(x=None, style=w, alpha=alpha)
        return x

    def grow(self):
        in_c, out_c = self.channels[self.now_growth], self.channels[self.now_growth+1] 
        self.model = UpBlock(in_c, out_c, self.style_dim, prev=self.model)
        self.now_growth += 1

        return self

##### Discriminator #####

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        self.now_growth = 1
        self.channels = channels

        self.model = DownBlock(channels[1], channels[0], next=None)

    def forward(self, x, alpha):
        return self.model(x=x, alpha=alpha)

    def grow(self):
        in_c, out_c = self.channels[self.now_growth+1], self.channels[self.now_growth] 
        self.model = DownBlock(in_c, out_c, next=self.model)
        self.now_growth += 1

        return self