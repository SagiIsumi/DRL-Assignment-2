import torch
import torch.nn as nn

class TD_learn():
    def __init__(self):
        feature_size=14
        self.model=nn.Sequential(
            nn.Linear(feature_size,28),
            nn.ReLU(),
            nn.Linear(28,56),
            nn.ReLU(),
            nn.Linear(56,28),
            nn.ReLU(),
            nn.Linear(28,feature_size),
            nn.ReLU(),
            nn.Linear(feature_size,1),
        )
        self.weights_func=torch.ones([feature_size])/100

    def greedy(board,epsilon):


    def eval(board):

    def selfplay(board):
