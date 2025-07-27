import torch
from config import *

class TriggerBotAutoencoderModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        assert capture_width == 256
        assert capture_height == 256

        self.conv1 = torch.nn.Conv2d(
            in_channels=3 * sample_window,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.fc1 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=3 * sample_window,
            kernel_size=5,
            stride=1,
            padding=2,
        )

    def encode(self, x):
        y = x

        y = self.conv1(y)
        y = torch.nn.functional.gelu(y)
        y = torch.nn.functional.max_pool2d(input=y, kernel_size=2, stride=2)

        y = self.conv2(y)
        y = torch.nn.functional.gelu(y)
        y = torch.nn.functional.max_pool2d(input=y, kernel_size=2, stride=2)

        y = self.conv3(y)
        y = torch.nn.functional.gelu(y)
        y = torch.nn.functional.max_pool2d(input=y, kernel_size=2, stride=2)

        y = y.view((len(y), -1))

        y = self.fc1(y)
        y = torch.nn.functional.gelu(y)

        return y

    def decode(self, x):
        y = x

        y = y.view((len(y), 64, 4, 4))

        y = torch.nn.functional.interpolate(input=y, scale_factor=4, mode="nearest")
        y = self.conv4(y)
        y = torch.nn.functional.gelu(y)

        y = torch.nn.functional.interpolate(input=y, scale_factor=4, mode="nearest")
        y = self.conv5(y)
        y = torch.nn.functional.gelu(y)

        y = torch.nn.functional.interpolate(input=y, scale_factor=4, mode="nearest")
        y = self.conv6(y)

        return y

    def forward(self, x):
        return self.decode(self.encode(x))
