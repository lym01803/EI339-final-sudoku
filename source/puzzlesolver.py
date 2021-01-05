import cv2
import numpy as np
import imutils
from imutils import perspective
from skimage.segmentation import clear_border
from model import *
from ctypes import *
import time


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

dclib = cdll.LoadLibrary("./dancinglink.dll")
puzzle_solver = dclib.solve
puzzle_solver.restype = c_int

def puzzle_func(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

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

label_to_number = [(i, i+1) for i in range(9)] + [(i, i-8) for i in range(9, 18)]
label_to_number = dict(label_to_number)
#print(label_to_number)
def cutting(puzzle, Model):
    global startTime
    Sudoku = [[0 for i in range(9)] for j in range(9)]
    H = puzzle.shape[0]
    W = puzzle.shape[1]
    pos = list()
    L = list()
    preprocess_imgs = [[None for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):
            x1 = i * W // 9
            x2 = (i+1) * W // 9
            y1 = j * H // 9
            y2 = (j + 1) * H // 9
            cell_img = puzzle[y1:y2, x1:x2]
            res, preprocess_imgs[j][i] = process_cell(cell_img, i, j)
            #cv2.imshow("w", cell_img)
            #cv2.waitKey(0)
            if not res is None:
                pos.append((j, i))
                L.append(res)
    #print("Cell Cut: ", time.time() - startTime)
    L = np.array(L)
    #print("loaded: ", time.time() - startTime)

    P, O, SO = Model.Predict(L)
    #print("Predicted: ", time.time() - startTime)
    for i in range(len(pos)):
        Sudoku[pos[i][0]][pos[i][1]] = label_to_number[P[i]]
        with open("../log/predict_report.txt", "a", encoding="utf8") as fout:
            fout.write("------------\n{}, {}\n{}\n{}\n".format(pos[i], P[i], O[i], SO[i]))
    return Sudoku, preprocess_imgs

def max_pool(img_in, size):
    img_out = img_in.copy()
    for x in range(img_in.shape[0] // size):
        for y in range(img_in.shape[1] // size):
            img_out[size*x : size*(x+1), size*y : size*(y+1)] = np.max(img_in[size*x : size*(x+1), size*y : size*(y+1)])
    return img_out

def process_cell(img, i, j):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img = max_pool(img, 5)
    img = cv2.resize(img, (28, 28))
    img = clear_border(img, 1)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #with open("../img_matrix.txt", "a", encoding="utf8") as fout:
    #    if (i, j) == (4, 1):
    #        fout.write("******************\n{}\n".format(img))
    #cv2.namedWindow("show", 0)
    #cv2.resizeWindow("show", 600, 600)
    #cv2.imshow("show", img)
    #cv2.waitKey(0)
    #print(img)
    p = np.sum(img) / (img.shape[0] * img.shape[1])
    if (p < 1):
        return None, img
    return img, img

def r_check(g, r):
    cnt = [0 for i in range(10)]
    for c in range(9):
        cnt[g[r][c]] += 1
    for i in range(1, 10):
        if cnt[i] > 1:
            return False
    return True 

def c_check(g, c):
    cnt = [0 for i in range(10)]
    for r in range(9):
        cnt[g[r][c]] += 1
    for i in range(1, 10):
        if cnt[i] > 1:
            return False
    return True

def a_check(g, a):
    r0 = a // 3 * 3
    c0 = a % 3 * 3
    cnt = [0 for i in range(10)]
    for i in range(3):
        for j in range(3): 
            cnt[g[r0+i][c0+j]] += 1
    for i in range(1, 10):
        if cnt[i] > 1:
            return False
    return True

def check_puzzle_valid(grid):
    for i in range(9):
        if not r_check(grid, i):
            return False
        if not c_check(grid, i):
            return False
        if not a_check(grid, i):
            return False
    return True

def solve_puzzle(res):
    if not check_puzzle_valid(res):
        return None
    
    arr_10_10 = (c_int * 10) * 10
    arr = arr_10_10()
    for i in range(9):
        for j in range(9):
            arr[i][j] = res[i][j]
        arr[i][9] = 0
    for j in range(10):
        arr[9][j] = 0
    
    state = puzzle_solver(arr)

    if state < 0:
        return None
    
    L = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):
            L[i][j] = arr[i][j]
    return L
    
def Show_in_terminal(grid):
    for i in range(9):
        if i % 3 == 0:
            print("+-------+-------+-------+")
        for j in range(9):
            if j % 3 == 0:
                print("|", end=" ")
            print(grid[i][j], end=" ")
        print("|")
    print("+-------+-------+-------+")

if __name__ == "__main__":
    global startTime
    startTime = time.time()
    print("Start: ", time.time() - startTime)
    img = cv2.imread("../test1/2-2.jpg", cv2.IMREAD_COLOR)
    puzzle = puzzle_func(img)
    if not puzzle is None:
        print("Puzzle Found: ", time.time() - startTime)
        res = cutting(puzzle)
        for i in range(9):
            for j in range(9):
                print(res[i][j], end=" ")
            print()
        print()
        if not res is None:
            solution = solve_puzzle(res)
            if solution is None:
                print("No solution found.")
            else:
                for i in range(9):
                    for j in range(9):
                        print(solution[i][j], end=" ")
                    print()
                print()
        print("Solved: ", time.time() - startTime)

