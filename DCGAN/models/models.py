import torch
from torch import nn


class GeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, height, width, activation_fn='relu'):

        super(GeneratorBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

        self.layer_norm = nn.LayerNorm(normalized_shape=[out_channels, height, width])

        if activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        if activation_fn == 'tanh':
            self.activation_fn = nn.Tanh()
        if activation_fn == 'sigmoid':
            self.activation_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.up_conv(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        return x


class GeneratorNetwork(nn.Module):

    def __init__(self, z_dim, image_channels=3):
        super(GeneratorNetwork, self).__init__()

        self.project_and_reshape = nn.Linear(in_features=z_dim, out_features=4*4*1024)

        self.up_conv_block_1 = GeneratorBlock(in_channels=1024, out_channels=512, kernel_size=2, stride=2, height=8, width=8)

        self.up_conv_block_2 = GeneratorBlock(in_channels=512, out_channels=256, kernel_size=2, stride=2, height=16, width=16)

        self.up_conv_block_3 = GeneratorBlock(in_channels=256, out_channels=128, kernel_size=2, stride=2, height=32, width=32)

        self.up_conv_block_4 = GeneratorBlock(in_channels=128, out_channels=image_channels, kernel_size=2, stride=2, height=64, width=64, activation_fn='sigmoid')

    def forward(self, x):

        x = self.project_and_reshape(x).reshape(-1, 1024, 4, 4)
        x = self.up_conv_block_1(x)
        x = self.up_conv_block_2(x)
        x = self.up_conv_block_3(x)
        x = self.up_conv_block_4(x)

        return x


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, height, width):
        super(DiscriminatorBlock, self).__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1))

        self.layer_norm = nn.LayerNorm(normalized_shape=[out_channels, height, width])

        self.activation_fn = nn.LeakyReLU()

    def forward(self, x):

        x = self.conv_block(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        return x


class DiscriminatorNetwork(nn.Module):

    def __init__(self, image_channels=3, kernel_size=3, stride=2, num_classes=1):
        super(DiscriminatorNetwork, self).__init__()

        self.conv_block_1 = DiscriminatorBlock(in_channels=image_channels, out_channels=128, kernel_size=kernel_size, stride=stride, height=32, width=32)

        self.conv_block_2 = DiscriminatorBlock(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride, height=16, width=16)

        self.conv_block_3 = DiscriminatorBlock(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride, height=8, width=8)

        self.conv_block_4 = DiscriminatorBlock(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=stride, height=4, width=4)

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)

        self.classifier = nn.Sequential(nn.Linear(in_features=1024*4*4, out_features=512),
                                        nn.ReLU(),
                                        nn.Linear(in_features=512, out_features=256),
                                        nn.ReLU(),
                                        nn.Linear(in_features=256, out_features=num_classes),
                                        nn.Sigmoid()
                                        )

    def forward(self, x):

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    tensor = torch.randn((10, 10))

    gen_net = GeneratorNetwork(10)

    output = gen_net(tensor)

    dis_net = DiscriminatorNetwork()

    classified_tensor = dis_net(output)


