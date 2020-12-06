import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class NetModel(nn.Module):
    def __init__(self, input_size, classes):
        super(NetModel, self).__init__()
        assert ((input_size - 8) % 4 == 0)
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 48, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_s = (input_size // 4 - 2)
        self.fc1 = nn.Linear(48 * self.fc_s * self.fc_s, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classes = classes
        self.fc3 = nn.Linear(1024, self.classes)
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.activate(self.dropout(self.conv1(x)))
        x = self.pool(self.activate(self.dropout(self.conv2(x))))
        x = self.pool(self.activate(self.dropout(self.conv3(x))))
        x = x.view(-1, 48 * self.fc_s * self.fc_s)
        x = self.activate(self.dropout(self.fc1(x)))
        x = self.activate(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class DigitClassifier():
    def __init__(self, size=28, classes=9):
        self.size = size
        self.classes = classes
        self.model = NetModel(self.size, self.classes) #RSNet([4, 4, 4], classes=10) #NetModel(self.size, self.classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))
        self.model.to(self.device)

    def Train(self, data, label):
        data = data.reshape(-1, 1, self.size, self.size)
        label = label.reshape(-1)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label, self.device), batch_size=32, shuffle=True)

        loss_list = []

        self.model.train()
        for (i, (data, label)) in enumerate(data_loader):
            out = self.model(data)
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        
        return loss_list

    def Test(self, data, label):
        data = data.reshape(-1, 1, self.size, self.size)
        label = label.reshape(-1)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label, self.device), batch_size=32, shuffle=True)

        loss_list = []
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for (data, label) in data_loader:
                out = self.model(data)
                loss = self.criterion(out, label)
                loss_list.append(loss.item())
                
                #print(out)
                #print(label)

                out_cpu = out.to("cpu").numpy()
                out_cpu = np.argmax(out_cpu, 1)
                label_cpu = label.to("cpu").numpy()
                
                #print(out_cpu)
                #print(label_cpu)

                total += data.shape[0]
                correct += np.sum(out_cpu == label_cpu)
                #print(total, correct)
        
        return loss_list, correct, total

    def Predict(self, data):
        data = data.reshape(-1, 1, self.size, self.size)
        data = torch.tensor(data, dtype=torch.float32)
        data = data / 255.0

        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            out = out.to("cpu").numpy()
        res = np.argmax(out, 1)
        return res

    def save(self, path, epoch):
        state = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(state, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        

class DataLabel(torch.utils.data.Dataset):
    def __init__(self, data, label, device):
        self.data = torch.tensor(data).to(device)
        self.label = torch.tensor(label).to(device)
    
    def __getitem__(self, index):
        data, target = self.data[index], self.label[index]
        return data, target
    
    def __len__(self):
        return len(self.data)

class RSNet(nn.Module):
    def __init__(self, layers, channels=1, size=28, classes=9):
        super(RSNet, self).__init__()
        self.size = size
        self.classes = classes
        self.in_channels = 16
        self.conv = nn.Conv2d(channels, self.in_channels, 5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.size = (self.size + 1) // 2
        self.layer1 = self.make_layer(self.in_channels, layers[0])
        self.layer2 = self.make_layer(self.in_channels * 2, layers[1], 2)
        self.size = (self.size + 1) // 2
        self.layer3 = self.make_layer(self.in_channels * 2, layers[2], 2)
        self.size = (self.size + 1) // 2
        self.fc = FcBlock(3, [self.in_channels * self.size * self.size, 256, 64, self.classes])

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.size * self.size * self.in_channels)
        x = self.fc(x)
        return x
        
    def make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout()
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        if self.downsample:
            residual = self.downsample(input)
        x = x + residual
        x = self.relu(x)
        x = self.dropout(x)
        return x

class FcBlock(nn.Module):
    def __init__(self, layers, dims):
        super(FcBlock, self).__init__()
        self.layers = []
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.layers.append(nn.Dropout())
            if i < layers - 1:
                self.layers.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.fc(x)
