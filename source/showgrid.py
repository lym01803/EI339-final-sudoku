import cv2
import numpy as np

num_img = []
red_num_img = []

board_img = cv2.imread("../imgs/board.png", cv2.IMREAD_COLOR)
board_img = cv2.resize(board_img, (56*9, 56*9))
for i in range(1, 10):
    num_img.append(cv2.imread("../imgs/{}.png".format(i), cv2.IMREAD_COLOR))
    red_num_img.append(cv2.imread("../imgs/{}r.png".format(i), cv2.IMREAD_COLOR))

def show_processed(processed, size=28, save_pth="../result/processed_result.png", showing=True):
    bg = np.zeros((size * 9, size * 9))
    bg = np.uint8(bg)
    for i in range(9):
        for j in range(9):
            bg[i*size : (i+1)*size, j*size : (j+1)*size] = processed[i][j][:, :]
    
    cv2.imwrite(save_pth, bg)
    if showing:
        cv2.imshow("processed_result", bg)
        cv2.waitKey(0)
    #print(bg)

def show_result(grid, mark, save_pth="../result/solve_result.png", showing=True):
    tmp_board = board_img.copy()
    for i in range(9):
        for j in range(9):
            if grid[i][j]:
                if mark[i][j]:
                    num = num_img[grid[i][j]-1]
                else:
                    num = red_num_img[grid[i][j]-1]
                tmp_board[(i*56+3):((i+1)*56-3), (j*56+3):((j+1)*56-3), :] = num
    cv2.imwrite(save_pth, tmp_board)
    if showing:
        cv2.imshow("solve_result", tmp_board)
        cv2.waitKey(0)
