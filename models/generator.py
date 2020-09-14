import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, initialSize, outputSize):
        super(Generator, self).__init__()
        outputChannel = 64
        self.generate = nn.Sequential(
            # input to output channel 1
            nn.ConvTranspose2d(initialSize, outputChannel * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(outputChannel * 16),
            # nn.ReLU(True),
            # output channel 1 to output channel 2
            # nn.ConvTranspose2d(
            #     outputChannel * 16, outputChannel * 8, 4, 2, 1, bias=False
            # ),
            nn.BatchNorm2d(outputChannel * 8),
            nn.ReLU(True),
            # output channel 2 to output channel 3
            nn.ConvTranspose2d(
                outputChannel * 8, outputChannel * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(outputChannel * 4),
            nn.ReLU(True),
            # output channel 3 to output channel 4
            nn.ConvTranspose2d(
                outputChannel * 4, outputChannel * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(outputChannel * 2),
            nn.ReLU(True),
            # output channel 4 to output channel 5
            nn.ConvTranspose2d(outputChannel * 2, outputChannel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(outputChannel),
            nn.ReLU(True),
            # output channel 5 to generator
            nn.ConvTranspose2d(outputChannel, outputSize, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.generate(input)
