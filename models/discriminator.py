import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, initialSize):
        super(Discriminator, self).__init__()
        outputChannel = 64
        self.shoot = nn.Sequential(
            # input to output channel 1
            nn.Conv2d(initialSize, outputChannel, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 1 to output channel 2
            nn.Conv2d(outputChannel, outputChannel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outputChannel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 2 to output channel 3
            nn.Conv2d(outputChannel * 2, outputChannel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outputChannel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 3 to output channel 4
            nn.Conv2d(outputChannel * 4, outputChannel * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outputChannel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 4 to output channel 5
            # nn.Conv2d(outputChannel * 8, outputChannel * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(outputChannel * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # output channel 5 to output channel 6
            # nn.Conv2d(outputChannel * 16, 1, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(outputChannel * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # output channel 6 to discriminator
            nn.Conv2d(outputChannel * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.shoot(input).view(-1, 1).squeeze(1)
