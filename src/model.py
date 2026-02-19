import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Conv. layers = letar efter mönster
        # Convolution layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        # Convolution layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Pooling = minskar bildstorlek
        self.pool = nn.MaxPool2d(2, 2)

        # Linear = fattar beslut
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        # Output = 10 scores (en per klass)

    def forward(self, x):

        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x