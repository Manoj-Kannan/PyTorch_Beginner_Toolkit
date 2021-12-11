# Load-Save:
# epoch: a measure of the number of times all of the training vectors are used once to update the weights.
# valid_loss_min: the minimum validation loss, this is needed so that when we continue the training, we can start with
# this rather than np.Inf value.
# state_dict: model architecture information. It includes the parameter matrices for each of the layers.
# optimizer: You need to save optimizer parameters especially when you are using Adam as your optimizer. Adam is an adaptive
# learning rate method, which means, it computes individual learning rates for different parameters which we would need if we
# want to continue our training from where we left off.

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()
