import numpy as np
from GridWorld import GridWorld
from configs import *
import argparse


class MontoCarlo():
    # simply generate episode randomly
    def __init__(self, max_ite=100):
        self.max_ite = max_ite  # maximum iterations
        self.env = GridWorld()  # assign the grid world as the environment

    def random_action(self):
        p = np.random.uniform(0,1)
        cumulated_p = 0
        for action, action_p in enumerate(self.env.policy):
            cumulated_p+=self.env.policy[action]
            if cumulated_p>=p:
                return action

    def take_action(self,x,y):
        action = self.random_action() # random policy
        nx = x + self.env.dirx[action]
        ny = y + self.env.diry[action]
        if nx < 0 or nx >= self.env.grid:  # can't go out of grids
            nx = x
        if ny < 0 or ny >= self.env.grid:
            ny = y
        return nx, ny

    def generate_episode(self):
        episode = list()
        x = np.random.randint(self.env.grid)
        y = np.random.randint(self.env.grid)
        state = [x, y]
        while True:
            x, y = state
            nx,ny = self.take_action(x,y)
            next_state = [nx, ny]
            episode.append(next_state)
            state = next_state
            if next_state in self.env.terminal:
                return episode

    def main_loop(self):
        N = np.zeros([self.env.grid, self.env.grid])
        for ite in range(0, self.max_ite):
            delta = 0
            episode = self.generate_episode()
            states = list()  # states have encounterd (for first time-stamp method)
            while len(episode) > 0:
                state = episode.pop(0)  # pop the head of the queue
                if state not in states and state not in self.env.terminal:
                    states.append(state)
                    x, y = state
                    N[x, y] += 1
                    # because V = V+(G-V)/N     R*(1+y+y^2+...+y^n) = R*(1-y^n)/(1-y)
                    if self.env.gamma == 1:
                        self.env.values[x, y] = self.env.values[x, y] + (self.env.reward * (len(episode)-1) -self.env.values[x, y])/N[x, y]
                    else:
                        self.env.values[x, y] = self.env.values[x, y] + (self.env.reward * (
                            1-self.env.gamma**(len(episode)-1))/(1-self.env.gamma)-self.env.values[x, y])/N[x, y]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ite', type=int, default=100,
                        help="maximum iterations")
    args = parser.parse_args()
    max_ite = args.ite

    MCtest = MontoCarlo(max_ite=max_ite)
    MCtest.main_loop()
    print("after {} iterations".format(MCtest.max_ite))
    MCtest.env.printValues()
