import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoDiscriminator(nn.Module):
    def __init__(self):
        super(VideoDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (3, 10), (1, 2))
        self.conv2 = nn.Conv2d(16, 8, (3, 8), (1, 2))
        self.conv3 = nn.Conv2d(8, 4, (3, 6), (1, 2))
        self.conv4 = nn.Conv2d(4, 1, (2, 6), (1, 2))

        self.fc = nn.Linear(27, 1)
        self.activ_fn = nn.Sigmoid()

    def __call__(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.activ_fn(x)
