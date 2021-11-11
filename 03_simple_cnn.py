# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=8,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),
                                 stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        # (out_channels * size of img after second pooling layer)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (28,28)
        x = self.pool(x)            # (14,14)
        x = F.relu(self.conv2(x))   # (14,14)
        x = self.pool(x)            # (7,7)
        x = x.reshape(x.shape[0], -1) # (batch_size, out_channels * (7,7))
        x = self.fc1(x)
        return x

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters of our neural network which depends on the dataset, and
# also just experimenting to see what works well (learning rate for example).
in_channels = 1 # ( 28x28 images)
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Training and Test data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
