import torch
import torch.nn as nn
import torch.nn.functional as F

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    '''
    Weighted Scaled Conv2d
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bias = self.conv.bias
        self.conv.bias = None       # the bias term should not be scaled
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5       # scaling function

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    '''
    See definition in the PixelNorm part of the important details file.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        # x has the shape of [N, C, H, W], to average over the channels we should choose dim=1

class ConvBlock(nn.Module):
    '''
    The 2 Conv 3*3 layers in the general architecture, mind that PixelNorm is only used in Generator layers.
    '''
    def __init__(self, in_channels, out_channels, use_pixelnorm=True) -> None:
        super().__init__()
        self.use_pixelnorm = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)        # the 2 Conv 3*3 layers
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pixelnorm else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pixelnorm else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3) -> None:
        super().__init__()
        self.initial_block = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),
            # the latent vector is shaped [N, z_dim, 1, 1], and will be changed to [N, in_channels, 4, 4], this should actually be a weighted-scaled ConvTranspose block
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),
            nn.LeakyReLU(),
            PixelNorm()
        )   # the initial block is a little different from the other blocks

        self.initial_to_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.to_rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_to_rgb])
        for i in range(len(factors) - 1):
            # change the proportion of the channels from factors[i] -> factors[i + 1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.to_rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))
        '''
        After the for loop, the prog_blocks and from_rgb_layers are like (when in_channels=512):
            prog_blocks             to_rgb_layers       img_size (set to 1024)
        0       512 -> 512              512 -> 3            [4, 4]
        1       512 -> 512              512 -> 3            [8, 8]
        2       512 -> 512              512 -> 3            [16, 16]
        3       512 -> 256              512 -> 3            [32, 32]
        4       256 -> 128              256 -> 3            [64, 64]
        5       128 -> 64               128 -> 3            [128, 128]
        6       64 -> 32                64 -> 3             [256, 256]
        7       32 -> 16                32 -> 3             [512, 512]
        8                               16 -> 3             [1024, 1024]
        '''
        

    def fade_in(self, alpha, upsampled, generated):
        '''
        See definition in the fade in part of the important details file.
        '''
        return torch.tanh(alpha * generated + (1 - alpha) * upsampled)

    def forward(self, x, alpha, steps):
        '''
        steps controls the resolution, steps = 0 returns image of [3, 4, 4], steps = 1 returns image of [3, 8, 8]...
        e.g. steps = 3, after the for loop, upsample has the shape of [c1, 32, 32] and [c2, 32, 32], so we need to use
        self.to_rgb_layers[3-1] and self.to_rgb_layers[3] to deal with the different channels.
        Mind that when steps = 1, upsample has the same channel number with the original self.initial_block(x), so we 
        should use self.initial_to_rgb to deal with it. As a result, we need to initialize self.to_to_rgb_layers with 
        it.
        '''
        out = self.initial_block(x)         # out has the shape of [in_channels, 4, 4]
        if steps == 0:
            return self.initial_to_rgb(out) # for the [4, 4] images, we do not apply the fade_in procedure

        for i in range(steps):
            upsample = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[i](upsample)
            # the upsample procedure is the nearest neighbor filter(interpolation)

        final_upsample_img = self.to_rgb_layers[steps - 1](upsample)   # use steps - 1 because the upsample procedure doesn't change the channels
        final_out_img = self.to_rgb_layers[steps](out)

        return self.fade_in(alpha, final_upsample_img, final_out_img)
        

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3) -> None:
        super().__init__()
        self.leaky = nn.LeakyReLU(0.2)
        self.prog_blocks, self.from_rgb_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.from_rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))
        '''
        After the for loop, the prog_blocks and from_rgb_layers are like (when in_channels=512):
            prog_blocks             from_rgb_layers     img_size (set to 1024)
        0       16 -> 32                3 -> 16             [1024, 1024]
        1       32 -> 64                3 -> 32             [512, 512]
        2       64 -> 128               3 -> 64             [256, 256]
        3       128 -> 256              3 -> 128            [128, 128]
        4       256 -> 512              3 -> 256            [64, 64]
        5       512 -> 512              3 -> 512            [32, 32]
        6       512 -> 512              3 -> 512            [16, 16]
        7       512 -> 512              3 -> 512            [8, 8]
        8                               3 -> 512            [4, 4]
        The from_rgb_layers has one more layer because we append the initial_from_rgb layer to the end
        '''

        self.initial_from_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.from_rgb_layers.append(self.initial_from_rgb)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)     # for the 0.5x downsample
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3),  # the + 1 is for the minibatch stddev part
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)    # use WSConv2d instead of the fully-connected layers in the paper
        )

    def fade_in(self, alpha, downsampled, discirminated):
        return alpha * discirminated + (1 - alpha) * downsampled

    def minibatch_std(self, x):
        # x should have the shape of [N, in_channels, H, W]
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)  # in_channels -> in_channels + 1

    def forward(self, x, alpha, steps):
        '''
        self.from_rgb_layers[0] and self.prog_blocks[0] are used for the final resolution image generated, 
        but that is for the largest steps, instead of steps=0, so we need to reverse the steps first
        '''
        reverse_steps = len(self.prog_blocks) - steps
        out = self.leaky(self.from_rgb_layers[reverse_steps](x))
        # e.g. steps = 0, then reverse_steps = 8, x: [3, 4, 4] -> out: [512, 4, 4]
        # e.g. stpes = 5, then reverse_steps = 3, x: [3, 128, 128] -> out: [128, 128, 128]

        if steps == 0:
            # the [4, 4] images don't have fade_in procedure (check with the Generator part)
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1) # return shape [N, 1]

        # according to the fade in logic (see more in important details.md), the downsample part should first
        # downsample and then apply from_rgb, the discriminated part should first apply from_rgb then downsample
        downsample = self.leaky(self.from_rgb_layers[reverse_steps + 1](self.downsample(x)))
        out = self.downsample(self.prog_blocks[reverse_steps](out)) # out has already done the from_rgb procedure

        out = self.fade_in(alpha, downsample, out)

        for i in range(reverse_steps + 1, len(self.prog_blocks)):
            out = self.prog_blocks[i](out)
            out = self.downsample(out)      # bring down the resolution

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

if __name__ == '__main__':
    from math import log2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    in_channels = 256
    gen = Generator(z_dim, in_channels, 3).to(device)
    dis = Discriminator(in_channels, 3).to(device)

    for img_size in [4, 8, 16, 32, 64, 128, 256]:
        # check it out with the case example in import details
        steps = int(log2(img_size / 4))
        x = torch.randn(4, z_dim, 1, 1).to(device)
        img = gen(x, 0.5, steps)
        assert img.shape == (4, 3, img_size, img_size)
        score = dis(img, 0.5, steps)
        assert score.shape == (4, 1)
        print('ok on {}'.format(img_size))