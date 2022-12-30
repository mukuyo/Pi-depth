import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms

#dataset directory
root = "./datasets/curve/"

epoch = 1

class Data(Dataset):
    def __init__(self, path):
        info = pd.read_csv(path)
        self.image_paths = info['image']
        self.labels = info['label']
        self.xmins = info['xmin']

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(root + path)

        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img).to(torch.float32)

        label = self.labels[index]
        image_path = self.image_paths[index]
        return img, label, image_path
    
    def __len__(self):
        print("image len:", len(self.image_paths))
        return len(self.image_paths)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(1, 16, 5)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*238*318, 4096)
        self.drop = nn.Dropout(0.5, inplace=False)
        self.fc2 = nn.Linear(4096, 49*30)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = x.view(-1, 7, 7, 5*2+20)

        return x

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        loss = outputs - targets

        return loss

if __name__ == "__main__":
    device = torch.device("cuda")

    # transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    dataset = Data(root + "pi_d-export.csv")

    # getitem and len
    dataloader = DataLoader(dataset, batch_size=1)

    # train model
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(epoch):
        for img, label, img_path in dataloader:
            img = img.to(device)
            outputs = net(img)
            
            loss = criterion(outputs, label)
            # loss.backward()
            print(outputs)

            break