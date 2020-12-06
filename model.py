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
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 12, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_s = (input_size // 4 - 2)
        self.fc1 = nn.Linear(12 * self.fc_s * self.fc_s, 240)
        self.fc2 = nn.Linear(240, 80)
        self.classes = classes
        self.fc3 = nn.Linear(80, self.classes)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.activate(self.conv1(x)))
        x = self.pool(self.activate(self.conv2(x)))
        x = x.view(-1, 12 * self.fc_s * self.fc_s)
        x = self.activate(self.dropout(self.fc1(x)))
        x = self.activate(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class DigitClassifier():
    def __init__(self, size=28, classes=9):
        self.size = size
        self.classes = classes
        self.model = NetModel(self.size, self.classes)
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
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, label)
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
