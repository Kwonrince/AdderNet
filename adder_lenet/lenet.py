import torch.nn as nn
import torch.nn.functional as F
from ladder import Ladder2D

class LeNet_add(nn.Module):
    def __init__(self):
        super(LeNet_add, self).__init__()
        self.conv1 = Ladder2D(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = Ladder2D(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(self.bn1(self.conv1(x)), (2,2))
        x = F.max_pool2d(self.bn2(self.conv2(x)), (2,2))
        x = x.reshape(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(-1, self.num_flat_features(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features