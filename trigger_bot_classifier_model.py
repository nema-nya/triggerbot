import torch


class TriggerBotClassifierModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=1024, out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=256)
        self.fc3 = torch.nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        y = x

        y = self.fc1(y)
        y = torch.nn.functional.gelu(y)

        y = self.fc2(y)
        y = torch.nn.functional.gelu(y)

        y = self.fc3(y)
        return y
