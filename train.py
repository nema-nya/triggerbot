import torch
import json
import numpy as np
from tqdm import tqdm as tqdm
import PIL.Image
import os
from trigger_bot_classifier_model import TriggerBotClassifierModel
from trigger_bot_autoencoder_model import TriggerBotAutoencoderModel
from config import *

assert torch.cuda.is_available()


class TriggerBotDataset(torch.utils.data.Dataset):

    def __init__(self, samples_file):
        with open(samples_file, "r") as file:
            self.samples = json.loads(file.read())
        self.miss_count = 0
        self.hit_count = 0
        for sample in self.samples:
            if sample["info"] is None:
                self.miss_count += 1
            else:
                self.hit_count += 1

    def get_weights(self):
        return torch.tensor([self.miss_count, self.hit_count]) / (
            self.hit_count + self.miss_count
        )

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
        frame_images = frame_images.permute((0, 3, 1, 2)).reshape((-1, capture_width, capture_height))
        return frame_images, 0 if info is None else 1

    def __len__(self):
        return len(self.samples)


def train_classifier(dataset, autoencoder_model):
    classifier_model = TriggerBotClassifierModel().cuda()
    classifier_optim = torch.optim.Adam(params=classifier_model.parameters())
    batch_size = 32
    weight = 1 / dataset.get_weights().cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    smooth_loss = 1.0
    smooth_accuracy = 0.0
    decay = 1e-1
    autoencoder_model.eval()
    classifier_model.train()
    for _ in range(epochs_count):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        with tqdm(loader) as enum:
            for x, y in enum:
                x = x.cuda()
                y = y.cuda()
                y_ = classifier_model(autoencoder_model.encode(x))
                loss = loss_fn(y_, y)
                classifier_optim.zero_grad()
                loss.backward()
                classifier_optim.step()
                accuracy = (y_.argmax(-1) == y).float().mean()
                smooth_loss = smooth_loss * (1 - decay) + loss.item() * decay
                smooth_accuracy = (
                    smooth_accuracy * (1 - decay) + accuracy.item() * decay
                )
                enum.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    smooth_loss=smooth_loss,
                    smooth_accuracy=smooth_accuracy,
                )
        torch.save(classifier_model.state_dict(), "classifier_checkpoint.pth")
    return classifier_model


def train_autoencoder(dataset):
    autoencoder_model = TriggerBotAutoencoderModel().cuda()
    autoencoder_optim = torch.optim.Adam(params=autoencoder_model.parameters())
    batch_size = 32
    loss_fn = torch.nn.MSELoss()
    smooth_loss = 1.0
    decay = 1e-1
    autoencoder_model.train()
    for _ in range(epochs_count):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        with tqdm(loader) as enum:
            for x, _ in enum:
                x = x.cuda()
                x_ = autoencoder_model(x)
                loss = loss_fn(x_, x)
                autoencoder_optim.zero_grad()
                loss.backward()
                autoencoder_optim.step()
                smooth_loss = smooth_loss * (1 - decay) + loss.item() * decay
                enum.set_postfix(
                    loss=loss.item(),
                    smooth_loss=smooth_loss,
                )
        torch.save(autoencoder_model.state_dict(), "autoencoder_checkpoint.pth")
    return autoencoder_model


def main():
    dataset = TriggerBotDataset("training_samples.json")
    autoencoder_model = train_autoencoder(dataset)
    train_classifier(dataset, autoencoder_model)


if __name__ == "__main__":
    main()
