import numpy as np
import argparse
import matplotlib.pyplot as plt
from configs import *


class GridWorld():
    def __init__(self, grid=4, gamma=1, terminal=[], reward=-1, policy=0.25*np.ones([4], dtype=np.float64)):
        self.grid = grid
        self.gamma = gamma
        self.terminal = terminal
        self.reward = reward
        self.policy = policy
        # if not assign, set the corners to terminal state
        if self.terminal == []:
            self.terminal.append([0, 0])
            self.terminal.append([self.grid-1, self.grid-1])
        # world map
        # init to zeros at first
        self.values = np.zeros([self.grid, self.grid])
        self.policies = np.zeros(
            [self.grid, self.grid, ACTION_NUM], dtype=bool)
        self.dirx = DIR_X
        self.diry = DIR_Y
        self.action_num = ACTION_NUM

    def printValues(self):
        print(self.values)
        plt.rcParams['axes.facecolor'] = 'white'
        plt.matshow(self.values, cmap=plt.cm.Greys)
        plt.colorbar()
        plt.title('The state value visulization')
        plt.show()
        
    def printPolicies(self):
        policies = ["" for _ in range(0,self.grid*self.grid)]
        for i in range(0,self.grid):
            for j in range(0,self.grid):
                for a in range(0,self.action_num):
                    if self.policies[i,j,a]:
                        policies[i*self.grid+j] += ACTIONS[a]
                    else:
                        policies[i*self.grid+j] += " "
        formated = ""
        for i in range(0,self.grid):
            for j in range(0,self.grid):
                formated+="| {} |".center(6)
            formated+='\n'
        print(formated.format(*policies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=4, help="the grid size")
    parser.add_argument('--gamma', type=float, default=1,
                        help="the discount for future expected rewards")
    parser.add_argument('--target', type=list, default=[],
                        help="the target position")
    parser.add_argument('--reward', type=float, default=-
                        1, help="the future reward")
    args = parser.parse_args()
    grid = args.grid
    gamma = args.gamma
    target = args.target
    reward = args.reward
    # if not assign, set the corners to target
    if target == []:
        target.append([0, 0])
        target.append([grid, grid])

    gridWorld = GridWorld()
    print(gridWorld.policy)
    print(gridWorld.values)
    # gridWorld.printPolicies()
    # gridWorld.printValues()
