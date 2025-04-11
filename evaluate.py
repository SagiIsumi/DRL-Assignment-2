from numba import njit
import numpy as np


@njit
def eval(board,score,action):
    if action==0:#up
        moved, new_board, new_score=_move(board,score,0,reverse=False)
    if action==1:#down
        moved, new_board, new_score=_move(board,score,0,reverse=True)
    if action==2:#left
        moved, new_board, new_score=_move(board,score,1,reverse=False)
    if action==3:#right
        moved, new_board, new_score=_move(board,score,1,reverse=True)
    return moved, new_board, new_score

@njit
def _move(board, score, axis, reverse):
    new_board = np.zeros_like(board)
    moved = False
    new_score = score

    for i in range(board.shape[axis]):
        line = board[:, i] if axis == 0 else board[i, :]
        if reverse:
            line = line[::-1]

        merged_line, line_score, line_moved = merge_line(line)
        if reverse:
            merged_line = merged_line[::-1]

        if axis == 0:
            new_board[:, i] = merged_line
        else:
            new_board[i, :] = merged_line

        if line_moved:
            moved = True
        new_score += line_score

    return moved, new_board, new_score

@njit
def merge_line(line):
    new_line = np.zeros_like(line)
    index = 0
    score = 0
    moved = False

    for i in range(len(line)):
        if line[i] != 0:
            if index > 0 and new_line[index - 1] == line[i]:
                new_line[index - 1] *= 2
                score += new_line[index - 1]
                moved = True
            else:
                new_line[index] = line[i]
                if i != index:
                    moved = True
                index += 1

    return new_line, score, moved

    