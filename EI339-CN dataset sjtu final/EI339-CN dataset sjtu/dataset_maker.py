import sys
import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

def Binary(pth, debug=False):
    img = cv2.imread(pth, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if debug:
        cv2.imshow("w", img)
        cv2.waitKey(0)
    return img

labels = [i+1 for i in range(10)]
types = ["training", "testing"]
DataSet = {"TrainData":{"data":[], "label":[]}, "TestData":{"data":[], "label":[]}}

for label in tqdm(labels):
    if label == 10:
        lb = 10
    else:
        lb = label + 10
    for tp in types:
        pth = "./{}/{}".format(label, tp)
        if tp == "training":
            sub = DataSet["TrainData"]
        else:
            sub = DataSet["TestData"]
        for rt, folds, files in os.walk(pth):
            for file in tqdm(files):
                filename = "{}/{}".format(pth, file)
                img = Binary(filename)
                if img.shape[0] == 28 and img.shape[1] == 28:
                    sub["data"].append(img)
                    sub["label"].append(lb)

DataSet["TrainData"]["data"] = np.vstack(DataSet["TrainData"]["data"])
DataSet["TrainData"]["data"] = DataSet["TrainData"]["data"].reshape(-1, 28, 28)
DataSet["TestData"]["data"] = np.vstack(DataSet["TestData"]["data"])
DataSet["TestData"]["data"] = DataSet["TestData"]["data"].reshape(-1, 28, 28)
DataSet["TrainData"]["label"] = np.array(DataSet["TrainData"]["label"])
DataSet["TestData"]["label"] = np.array(DataSet["TestData"]["label"])

with open("../../EI339DataSet/DataSet.pkl", "wb") as fout:
    pickle.dump(DataSet, fout)
