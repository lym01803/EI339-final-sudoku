import cv2
import numpy as np
import imutils
from imutils import perspective
from skimage.segmentation import clear_border
from model import *

def puzzle(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #cv2.imshow("window", thresh)
    #cv2.waitKey(0)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_contours = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            puzzle_contours = approx
            break
    
    if not puzzle_contours is None:
        #print(puzzle_contours)
        puzzle = perspective.four_point_transform(image, puzzle_contours.reshape(4, 2))
        #cv2.imshow("window", puzzle)
        #cv2.waitKey(0)
        return puzzle
    else:
        return None

def cutting(puzzle):
    Sudoku = [[0 for i in range(9)] for j in range(9)]
    H = puzzle.shape[0]
    W = puzzle.shape[1]
    pos = list()
    L = list()
    for i in range(9):
        for j in range(9):
            x1 = i * W // 9
            x2 = (i+1) * W // 9
            y1 = j * H // 9
            y2 = (j + 1) * H // 9
            cell_img = puzzle[y1:y2, x1:x2]
            res = process_cell(cell_img)
            #cv2.imshow("w", cell_img)
            #cv2.waitKey(0)
            if not res is None:
                pos.append((j, i))
                L.append(res)

    L = np.array(L)
    Model = DigitClassifier(28, 10)
    Model.load("./trained_models/RS18_09model.pth")
    print("loaded")

    P = Model.Predict(L)
    print("Predicted")

    for i in range(len(pos)):
        Sudoku[pos[i][0]][pos[i][1]] = P[i]
    
    return Sudoku

def process_cell(img):
    img = cv2.resize(img, (28, 28))
    img = clear_border(img, 2)
    #cv2.imshow("w", img)
    #cv2.waitKey(0)
    p = np.sum(img > 0) / (img.shape[0] * img.shape[1])
    if (p < 0.05):
        return None
    return img

if __name__ == "__main__":
    img = cv2.imread("./sudoku_puzzle.jpg", cv2.IMREAD_COLOR)
    puzzle = puzzle(img)
    if not puzzle is None:
        res = cutting(puzzle)
        for i in res:
            for j in i:
                print(j, end=" ")
            print()
