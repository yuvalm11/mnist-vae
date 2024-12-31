import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # 28x28x1 -> 24x24x1 -> 12x12x2 -> 10x10x4 -> 5x5x4 -> 100 -> 10
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, 2)

        x = x.view(-1, 100)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)

        return x