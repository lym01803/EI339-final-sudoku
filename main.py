from model import *
import numpy as np
import torchvision
import torch
from PIL import Image

train_dataset = torchvision.datasets.MNIST(".", train=True, download=False)
test_dataset = torchvision.datasets.MNIST(".", train=False, download=False)

def OnehotLabel(L, classes):
    l = []
    for i in L:
        tmp = [1 if x==i else 0 for x in range(classes)]
        l.append(tmp.copy())
    return np.vstack((l))

train_data = train_dataset.data
train_label = train_dataset.targets
test_data = test_dataset.data
test_label = test_dataset.targets

Model = DigitClassifier(28, 10)
save_pth = "./09model.pth"
for i in range(100):
    train_loss = Model.Train(train_data, train_label)
    print(i, sum(train_loss) / len(train_loss))
    with open("./train_loss.txt", "w", encoding="utf8") as fout:
        for loss in train_loss:
            fout.write("{}\n".format(loss))
    
    if (i + 1) % 10 == 0:
        test_loss, c, t = Model.Test(test_data, test_label)
        print("test, {} / {}, Accuracy: {}".format(c, t, 1.0*c/t))
        with open("./test_loss.txt", "w", encoding="utf8") as fout:
            for loss in test_loss:
                fout.write("{}\n".format(loss))
        print("Saving...")
        Model.save(path=save_pth, epoch=i)

