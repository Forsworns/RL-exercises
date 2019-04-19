import numpy as np
import argparse
from CliffWorld import CliffWorld

# pay attention that x is the row and y is the column
'''
   0 -------> y
    |
    |
   \|/ x
'''


class QL:
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
            while state != self.env.goal:
                x, y = state
                action = self.epsilonGreedy(state)
                nx, ny, reward = self.takeAction(x, y, action)
                actionValue = self.env.actionValue[:, nx, ny]
                greedyAction = np.argmax(actionValue)
                self.env.actionValue[action, x, y] = self.env.actionValue[action, x, y]+self.alpha*(
                    reward+self.env.gamma*self.env.actionValue[greedyAction, nx, ny]-self.env.actionValue[action, x, y])
                state = [nx, ny]


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

    ql = QL(max_ite, alpha, epsilon,
            gridRow, gridCol, rCliff, rNormal)
    ql.main_loop()
    print("after {} iterations".format(ql.max_ite))
    ql.env.printPolicies()
    ql.env.printValues()
