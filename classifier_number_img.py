from model import *
import sys
import cv2
import imutils

img = cv2.imread("./tmp.png")
img = cv2.resize(img, (28, 28))

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
cv2.imshow("window", img)
cv2.waitKey(0)

Model = DigitClassifier(28, 10)
print("loading...")
Model.load("./RS18_09model-80epoch.pth")
print("loaded")

P = Model.Predict(np.array([img]))
print(P)
