import numpy as np
import argparse
from CliffWorld import CliffWorld
from configs import *

# pay attention that x is the row and y is the column


class Sarsa:
    def __init__(self, maxIte, alpha, epsilon, gridRow, gridCol, rCliff, rNormal, verbose=True):
        self.env = CliffWorld(gridRow, gridCol, rCliff, rNormal)
        self.max_ite = maxIte
        self.alpha = alpha
        self.epsilon = epsilon
        self.verbose = verbose

    def epsilonGreedy(self, state):
        rand = np.random.rand()
        # seperate the selection of the best action into two part
        if rand < 1-self.epsilon/self.env.actionNum:
            x, y = state
            actionValue = self.env.actionValue[:, x, y]
            greedyAction = np.argmax(actionValue)
            return greedyAction
        else:
            action = np.random.randint(self.env.actionNum)
            return action

    def takeAction(self, x, y, action):
        nx = x + self.env.dirx[action]
        ny = y + self.env.diry[action]

        # in cliff world won't go out of the map
        if ny < 0 or ny >= self.env.gridCol or nx < 0 or nx >= self.env.gridRow:  # can't go out of grids
            reward = self.env.rNormal
            return x, y, reward

        if self.env.gridWorld[nx, ny]:
            nx, ny = self.env.start  # back to start point
            reward = self.env.rCliff
        else:
            reward = self.env.rNormal
        return nx, ny, reward

    def main_loop(self):
        for ite in range(0, self.max_ite):
            if self.verbose:
                print("ite is", ite)
            state = self.env.start
            action = self.epsilonGreedy(state)
            while state != self.env.goal:
                x, y = state
                nx, ny, reward = self.takeAction(x, y, action)
                a = action
                action = self.epsilonGreedy(state)
                self.env.actionValue[a, x, y] = self.env.actionValue[a, x, y]+self.alpha*(
                    reward+self.env.gamma*self.env.actionValue[action, nx, ny]-self.env.actionValue[a, x, y])
                state = [nx, ny]
                #print("a is",a)
                # print(self.env.actionValue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ite', type=int, default=2000,
                        help="maximum iterations")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="step size")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="exploration rate")
    parser.add_argument('--gridRow', type=int, default=4,
                        help="the gridRow size")
    parser.add_argument('--gridCol', type=int, default=4,
                        help="the gridCol size")
    parser.add_argument('--rCliff', type=float, default=-
                        100, help="the future reward for Cliff grids")
    parser.add_argument('--rNormal', type=float, default=-
                        1, help="the future reward for normal grids")
    args = parser.parse_args()
    gridRow = args.gridRow
    gridCol = args.gridCol
    rCliff = args.rCliff
    rNormal = args.rNormal
    max_ite = args.ite
    alpha = args.alpha
    epsilon = args.epsilon

    sarsa = Sarsa(max_ite, alpha, epsilon, gridRow, gridCol, rCliff, rNormal)
    sarsa.main_loop()
    print("after {} iterations".format(sarsa.max_ite))
    sarsa.env.printPolicies()
    sarsa.env.printValues()
