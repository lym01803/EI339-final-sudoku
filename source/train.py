from model import *
import numpy as np
import torchvision
import torch
from PIL import Image
import sys
import pickle

train_dataset = torchvision.datasets.MNIST("..", train=True, download=False)
test_dataset = torchvision.datasets.MNIST("..", train=False, download=False)

def OnehotLabel(L, classes):
    l = []
    for i in L:
        tmp = [1 if x==i else 0 for x in range(classes)]
        l.append(tmp.copy())
    return np.vstack((l))

def report_accuracy(c, t):
    ty = ["1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    for i in range(18):
        print("{} : {} / {} , {:.2f}%".format(ty[i], c[i], t[i], c[i]/t[i]*100))

#整合两个不同来源的数据集
idx = (train_dataset.targets != 0)
train_data1 = train_dataset.data[idx]
train_label1 = train_dataset.targets[idx]
idx = (test_dataset.targets != 0)
test_data1 = test_dataset.data[idx]
test_label1 = test_dataset.targets[idx]

with open("../EI339DataSet/DataSet_1.pkl", "rb") as fin:
    dataset2 = pickle.load(fin)

train_data2 = torch.from_numpy(dataset2["TrainData"]["data"])
train_label2 = torch.from_numpy(dataset2["TrainData"]["label"])
test_data2 = torch.from_numpy(dataset2["TestData"]["data"])
test_label2 = torch.from_numpy(dataset2["TestData"]["label"])

train_data = torch.cat((train_data1, train_data2))
test_data = torch.cat((test_data1, test_data2))
train_label = torch.cat((train_label1, train_label2)) - 1
test_label = torch.cat((test_label1, test_label2)) - 1

print("Data Set Size:\n", train_data.shape, train_label.shape, test_data.shape, test_label.shape)

Model = DigitClassifier(size=28, classes=18, method="RSNet18")
save_pth = "../trained_models/RSNet18_2021.pth"

start = 0
if (len(sys.argv) >= 3) and sys.argv[1] == "load":
    state = Model.load(sys.argv[2])
    start = state["epoch"] + 1
    print("learning rate: {}, epoch: {}".format(Model.lr, state["epoch"]))

for i in range(start, 100):
    train_loss, train_acc = Model.Train(train_data, train_label)
    print("epoch:{}  loss:{}  accuracy:{:.2f}%".format(i, sum(train_loss) / len(train_loss), train_acc*100))
    with open("../log/train_loss_rs18_new.txt", "w", encoding="utf8") as fout:
        for loss in train_loss:
            fout.write("{}\n".format(loss))
    with open("../log/train_epoch_average_loss_rs18_new.txt", "a", encoding="utf8") as fout:
        fout.write("{}, {}, {}\n".format(i, sum(train_loss) / len(train_loss), train_acc))
    if (i + 1) % 10 == 0:
        test_loss, c, t = Model.Test(test_data, test_label)
        report_accuracy(c, t)
        with open("../log/test_loss_rs18_new.txt", "a", encoding="utf8") as fout:
            for loss in test_loss:
                fout.write("{}\n".format(loss))
        print("Saving...")
        Model.save(path="../trained_models/RSNet18_2021_epoch{}.pth".format(i+1), epoch=i)
