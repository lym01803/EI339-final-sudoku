import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class SimpleNet(nn.Module):
    def __init__(self, input_size, classes):
        super(SimpleNet, self).__init__()
        assert ((input_size - 8) % 4 == 0)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv3 = nn.Conv2d(16, 48, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_s = (input_size // 4 - 2)
        self.fc1 = nn.Linear(16 * self.fc_s * self.fc_s, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classes = classes
        self.fc3 = nn.Linear(84, self.classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * self.fc_s * self.fc_s)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class DigitClassifier():
    def __init__(self, size, classes, method="RSNet18"):
        self.size = size
        self.classes = classes
        if method == "RSNet18":
            self.model = RSNet18(1, size, classes) 
        elif method == "SimpleNet":
            self.model = SimpleNet(self.size, self.classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))
        self.model.to(self.device)
        self.train_count = 0

    def Train(self, data, label, half = 20):
        self.train_count += 1
        data = data.reshape(-1, 1, self.size, self.size)
        label = label.reshape(-1)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label, self.device), batch_size=64, shuffle=True)

        loss_list = []
        total = 0
        correct = 0
        self.model.train()
        for (i, (data, label)) in enumerate(data_loader):
            out = self.model(data)
            out_label = torch.argmax(out, 1)
            correct += torch.sum(out_label == label)
            total += out.shape[0]
            loss = self.criterion(out, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        
        if self.train_count % 25 == 0:
            self.lr *= 0.5
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        correct = correct.to("cpu").numpy()
        return loss_list, correct / total

    def Test(self, data, label):
        data = data.reshape(-1, 1, self.size, self.size)
        label = label.reshape(-1)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label, self.device), batch_size=64, shuffle=True)

        loss_list = []
        correct = [0 for i in range(20)]
        total = [0 for i in range(20)]
        self.model.eval()
        with torch.no_grad():
            for (data, label) in data_loader:
                out = self.model(data)
                loss = self.criterion(out, label)
                loss_list.append(loss.item())
            
                out_cpu = out.to("cpu").numpy()
                out_cpu = np.argmax(out_cpu, 1)
                label_cpu = label.to("cpu").numpy()
                
                eq = (out_cpu == label_cpu)
                for cl in range(20):
                    correct[cl] += np.sum(eq[label_cpu == cl])
                    total[cl] += label_cpu[label_cpu == cl].shape[0]

        return loss_list, correct, total

    def Predict(self, data):
        data = data.reshape(-1, 1, self.size, self.size)
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        data = data / 255.0

        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            softmax_out = F.softmax(out)
            out = out.to("cpu").numpy()
            softmax_out = softmax_out.to("cpu").numpy()

        res = np.argmax(out, 1)
        return res, out, softmax_out

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
        self.lr = self.optimizer.param_groups[0]["lr"]
        return {"epoch": checkpoint["epoch"]}

class DataLabel(torch.utils.data.Dataset):
    def __init__(self, data, label, device):
        self.data = torch.tensor(data).to(device)
        self.label = torch.tensor(label).to(device)
    
    def __getitem__(self, index):
        data, target = self.data[index], self.label[index]
        return data, target
    
    def __len__(self):
        return len(self.data)

class RSNet18(nn.Module):
    def __init__(self, channels=1, size=28, classes=9):
        super(RSNet18, self).__init__()
        self.channels = 64
        self.conv1 = nn.Conv2d(channels, self.channels, 3, stride=1, padding=1) #1
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self.make_layer(self.channels, 2, 1) #4 64
        self.layer2 = self.make_layer(self.channels * 2, 2, 2) #4 128
        self.layer3 = self.make_layer(self.channels * 2, 2, 2) #4 256
        self.layer4 = self.make_layer(self.channels * 2, 2, 2) #4 512
        self.GAP = nn.AdaptiveAvgPool2d((1, 1)) # Batch_size * self.channels * 1 * 1
        self.fc = FcBlock(2, [self.channels, 1000, classes]) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.GAP(x)
        x = x.view(-1, self.channels)
        x = self.fc(x)
        return x
        
    def make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResBlock(self.channels, out_channels, stride, downsample))
        self.channels = out_channels
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.bn2(self.conv2(x))
        if self.downsample:
            residual = self.downsample(input)
        x = x + residual
        x = self.relu(x)
        return x

class FcBlock(nn.Module):
    def __init__(self, layers, dims):
        super(FcBlock, self).__init__()
        self.layers = []
        self.relu = nn.LeakyReLU(inplace=False)
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < layers - 1:
                self.layers.append(nn.Dropout())
                self.layers.append(self.relu)
        self.fc = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.fc(x)
