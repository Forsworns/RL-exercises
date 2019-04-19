import numpy as np
from GridWorld import GridWorld
from PE import PolicyEvaluation
from configs import *
import argparse


class PolicyImprovement():
    # use greedy method to improve the policies in each state
    def __init__(self, max_ite=400):
        self.iterations = max_ite  # record the total iterations
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
        for ite in range(0, self.max_ite):
            values = self.env.values
            self.env.values = np.zeros([self.env.grid, self.env.grid])
            policies = self.env.policies # use np.copy is more safe
            self.env.policies = np.zeros([self.env.grid, self.env.grid, self.env.action_num], dtype=bool)
            # (i,j) is a state in fact
            for i in range(0, self.env.grid):
                for j in range(0, self.env.grid):
                    if [i, j] not in self.env.terminal:
                        action_value = np.zeros([self.env.action_num])
                        for a in range(0, self.env.action_num):
                            ni,nj = self.take_action(i,j,a)
                            action_value[a] = values[ni, nj]
                        max_value = np.max(action_value)
                        greedy_actions = list()
                        [greedy_actions.append(a) for a in range(0, self.env.action_num) if action_value[a]==max_value] 
                        # have found the greedy policy for state (i,j)
                        for a in greedy_actions:
                            self.env.policies[i, j, a] = True
                            ni,nj = self.take_action(i,j,a)
                            self.env.values[i, j] = self.env.values[i, j] + \
                                self.env.reward+self.env.gamma*values[ni, nj]
                        self.env.values[i, j] /= len(greedy_actions)
            if self.policies_stable(policies):
                self.iterations = ite
                break

    def policies_stable(self, policies):
        return (self.env.policies == policies).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.001, help="the threshold to stop the iteration")
    parser.add_argument('--ite', type=int, default=400,
                        help="maximum iterations")
    args = parser.parse_args()
    delta = args.delta
    max_ite = args.ite

    PItest = PolicyImprovement(max_ite=max_ite)
    PItest.main_loop()
    print("converge after {} iterations".format(PItest.iterations))
    PItest.env.printValues()
    PItest.env.printPolicies()
    
