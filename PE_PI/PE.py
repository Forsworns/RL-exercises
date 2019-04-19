import numpy as np
from GridWorld import GridWorld
from configs import *
import argparse


class PolicyEvaluation():
    # simply apply the specific policy iteratively
    def __init__(self, delta=0.001, max_ite=400):
        self.iterations = max_ite  # record the total iterations
        self.delta = delta  # threshold for iteration
        self.max_ite = max_ite  # maximum iterations
        self.env = GridWorld()  # assign the grid world as the environment

    def take_action(self,i,j,a):
        ni = i+self.env.dirx[a]
        nj = j+self.env.diry[a]
        if ni < 0 or ni >= self.env.grid:
            ni = i
        if nj < 0 or nj >= self.env.grid:
            nj = j
        return ni,nj

    def main_loop(self):
        delta = INFINITE
        for ite in range(0, self.max_ite):
            values = self.env.values # use np.copy is more safe
            self.env.values = np.zeros([self.env.grid, self.env.grid])
            # (i,j) is a state in fact
            for i in range(0, self.env.grid):
                for j in range(0, self.env.grid):
                    if [i, j] not in self.env.terminal:
                        for a in range(0, self.env.action_num):
                            ni,nj = self.take_action(i,j,a)
                            self.env.values[i, j] = self.env.values[i, j] + self.env.policy[a] * \
                                (self.env.reward+self.env.gamma*values[ni, nj])
                        # calculate the change of the value function
                        delta = min(delta, abs(
                            self.env.values[i, j]-values[i, j]))
            if delta < self.delta:
                self.iterations = ite
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.001, help="the threshold to stop the iteration")
    parser.add_argument('--ite', type=int, default=400,
                        help="maximum iterations")
    args = parser.parse_args()
    delta = args.delta
    max_ite = args.ite

    PEtest = PolicyEvaluation(delta=delta,max_ite=max_ite)
    PEtest.main_loop()
    print("converge after {} iterations".format(PEtest.iterations))
    PEtest.env.printValues()
