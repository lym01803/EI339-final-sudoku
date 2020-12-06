import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NetModel(nn.Module):
    def __init__(self, input_size, classes):
        super(NetModel, self).__init__()
        assert ((input_size - 12 % 4) == 0)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_s = (input_size // 4 - 3)
        self.fc1 = nn.Linear(12 * self.fc_s * self.fc_s, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classes = classes
        self.fc3 = nn.Linear(84, self.classes)
        self.activate = F.leaky_relu

    def forward(self, x):
        x = self.pool(self.activate(self.conv1(x)))
        x = self.pool(self.activate(self.conv2(x)))
        x = x.view(-1, 12 * self.fc_s * self.fc_s)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.fc3(x)
        return nn.Softmax(x)

class DigitClassifier():
    def __init__(self, size=28, classes=9):
        self.size = size
        self.classes = classes
        self.model = NetModel(self.size, self.classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.params, lr=0.001)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, data, label):
        data = data.reshape(-1, self.size, self.size, 1)
        label = label.reshape(-1)
        data = data.astype(np.float32)
        label = label.astype(np.float32)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label), batch_size=32, shuffle=True)

        loss_list = []

        self.model.train()
        for (data, label) in data_loader:
            out = self.model(data)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        
        return loss_list

    def Test(self, data, label):
        data = data.reshape(-1, self.size, self.size, 1)
        label = label.reshape(-1)
        data = data.astype(np.float32)
        label = label.astype(np.float32)
        data = data / 255.0

        data_loader = torch.utils.data.DataLoader(dataset=DataLabel(data, label), batch_size=32, shuffle=True)

        loss_list = []

        self.model.eval()
        for (data, label) in data_loader:
            out = self.model(data)
            loss = self.criterion(out, label)
            loss_list.append(loss.item())
        
        return loss_list

    def Predict(self, data):
        data = data.reshape(-1, self.size, self.size, 1)
        data = data.astype(np.float32)
        data = data / 255.0

        self.model.eval()
        out = self.model(data)

        out = out.to("cpu").numpy()
        res = np.argmax(out, 1)
        return res

class DataLabel(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data)
        self.label = torch.tensor(label)
    
    def __getitem__(self, index):
        data, target = self.data[index], self.label[index]
        return data, target
    
    def __len__(self):
        return len(self.data)
