import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(3, 16, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

if __name__ == "__main__":
    net = Net()

    device = torch.device("cuda")
    net = net.to(device)

    Epoch = 1
    for i in range(Epoch):
        # outputs = net(inputs)
        print("OK")