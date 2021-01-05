from puzzlesolver import *
from showgrid import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--model", type=str, default="RSNet18")
parser.add_argument("--model_path", type=str, default="../trained_models/RSNet18_2021.pth")
args = parser.parse_args()

img = cv2.imread(args.img, cv2.IMREAD_COLOR)
puzzle = puzzle_func(img)
start_time = time.time()
Model = DigitClassifier(28, 18, method=args.model)
Model.load(args.model_path)
load_time = time.time()
print("Load Model. Time Consuming: {:.3f} s".format(load_time - start_time))
if not puzzle is None:
    res, processed = cutting(puzzle, Model)
    show_processed(processed)
    found_time = time.time()
    if not res is None:
        print("Recognized Result:")
        Show_in_terminal(res)
        rec_time = time.time()
        print("Recognize Puzzle. Time Consuming: {:.3f} s".format(rec_time - load_time))
        mark = [[res[i][j] != 0 for j in range(9)] for i in range(9)]
        print(mark)
        solution = solve_puzzle(res)
        if solution is None:
            print("No solution found.")
            show_result(res, mark)
        else:
            Show_in_terminal(solution)
            show_result(solution, mark)
        solve_time = time.time()
        print("Solve Sudoku. Time Consuming: {:.3f} s".format(solve_time - rec_time))
end_time = time.time()
print("Total Time Consuming: {:.3f} s".format(end_time - start_time))

