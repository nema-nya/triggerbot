import torch
import json
import numpy as np
from tqdm import tqdm as tqdm
import PIL.Image
import os

assert torch.cuda.is_available()


class TriggerBotDataset(torch.utils.data.Dataset):

    def __init__(self, samples_file):
        with open(samples_file, "r") as file:
            self.samples = json.loads(file.read())

    def __getitem__(self, index):
        frames = self.samples[index]["frames"]
        info = self.samples[index]["info"]
        frame_images = []
        for frame in frames:
            image = PIL.Image.open(os.path.join("outputs", frame))
            image = torch.tensor(np.array(image))
            image = image[:, :, :3].float() / 255
            frame_images.append(image)
        frame_images = torch.stack(frame_images)
        frame_images = frame_images.permute((0, 3, 1, 2)).reshape((-1, 512, 512))
        return frame_images, 0 if info is None else 1

    def __len__(self):
        return len(self.samples)


class TriggerBotModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=15,
            out_channels=64,
            kernel_size=7,
            stride=4,
            padding=3,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=7,
            stride=4,
            padding=3,
        )
        self.conv3 = torch.nn.Conv2d(
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
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=4,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.fc1 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        y = x

        y = self.conv1(y)
        y = torch.nn.functional.gelu(y)

        y = self.conv2(y)
        y = torch.nn.functional.gelu(y)

        y = self.conv3(y)
        y = torch.nn.functional.gelu(y)

        y = self.conv4(y)
        y = torch.nn.functional.gelu(y)

        y = y.view((len(y), 256))

        y = self.fc1(y)
        y = torch.nn.functional.gelu(y)

        y = self.fc2(y)
        return y


def main():
    dataset = TriggerBotDataset("training_samples.json")
    model = TriggerBotModel().cuda()
    optim = torch.optim.Adam(params=model.parameters())
    epochs_count = 10
    batch_size = 32
    with tqdm(range(epochs_count)) as enum:
        for _ in enum:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )
            for x, y in loader:
                x = x.cuda()
                y = y.cuda()
                y_ = model(x)
                loss = torch.nn.functional.cross_entropy(y_, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                enum.set_postfix(loss=loss.item())


if __name__ == "__main__":
    main()
