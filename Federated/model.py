import torch
from torch import nn
import torch.nn.functional as F


def convrelu(in_channels, out_channels, kernel_size, padding, pooling):
    """
    This function creates a sequence of convolutional, ReLU, and max pooling layers.

    Parameters:
    - in_channels (int): The number of input channels for the convolutional layer.
    - out_channels (int): The number of output channels for the convolutional layer.
    - kernel_size (int): The size of the convolutional kernel.
    - padding (int): The amount of zero-padding added to the input.
    - pooling (int): The size of the max pooling window.

    Returns: 
    - nn.Sequential: A sequence of convolutional, ReLU, and max pooling layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pooling, padding=0, dilation=1, stride=pooling,
                     return_indices=False, ceil_mode=False)
    )


def convreluBlock(In_channels, Out_channels, kernel_size, padding):
    """
    This function creates a sequence of transposed convolutional and ReLU layers.
    It is used to upsample the feature maps in a neural network for image generation tasks.

    Parameters:
    - In_channels (int): The number of input channels for the transposed convolutional layer.
    - Out_channels (int): The number of output channels for the transposed convolutional layer.
    - kernel_size (int): The size of the transposed convolutional kernel.
    - padding (int): The amount of zero-padding added to the input.

    Returns:
    - nn.Sequential: A sequence of transposed convolutional and ReLU layers.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(In_channels, Out_channels,
                           kernel_size, padding=padding, stride=2),
        nn.ReLU(inplace=True)
    )


class RadioNet(nn.Module):
    def __init__(self, d_in=2):
        super(RadioNet, self).__init__()
        self.d_in = d_in
        channels = 8 if d_in <= 3 else 10

        self.embeds = nn.Sequential(convrelu(d_in, channels, 3, 1, 1),
                                    convrelu(channels, 40, 5, 2, 2),
                                    convrelu(40, 50, 5, 2, 2),
                                    convrelu(50, 60, 5, 2, 1),
                                    convrelu(60, 100, 5, 2, 2),
                                    convrelu(100, 100, 3, 1, 1),
                                    convrelu(100, 150, 5, 2, 2),
                                    convrelu(150, 300, 5, 2, 2),
                                    convrelu(300, 500, 5, 2, 2))

        self.decoders = nn.Sequential(convreluBlock(500, 300, 4, 1),
                                      convreluBlock(300*2, 150, 4, 1),
                                      convreluBlock(150*2, 100, 4, 1),  #
                                      convrelu(100*2, 100, 3, 1, 1),
                                      convreluBlock(100*2, 60, 6, 2),
                                      convrelu(60*2, 50, 5, 2, 1),
                                      convreluBlock(50*2, 40, 6, 2),
                                      convreluBlock(40*2, 20, 6, 2),
                                      convrelu(20+channels+d_in, 20, 5, 2, 1),
                                      convrelu(20+d_in, 1, 5, 2, 1))

    def forward(self, x):
        embed_outputs = []
        # x = x[:, 0:self.d_in, :, :]
        embed_outputs.append(x)
        for layer in self.embeds:
            x = layer(x)
            embed_outputs.append(x)
        embed_outputs.reverse()

        for i, layer in enumerate(self.decoders):
            if i == 0:
                x = layer(embed_outputs[i])
            elif i < 8:
                x = torch.cat([x, embed_outputs[i]], dim=1)
                x = layer(x)
            elif i == 8:
                x = torch.cat([x, embed_outputs[i]], dim=1)
                x = torch.cat([x, embed_outputs[i+1]], dim=1)
                x = layer(x)
            else:
                x = torch.cat([x, embed_outputs[i]], dim=1)
                x = layer(x)
        return x
