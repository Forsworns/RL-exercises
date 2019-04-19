import numpy as np
from GridWorld import GridWorld
from configs import *
import argparse


class TemporalDifference():
    # TD(0)
    def __init__(self, max_ite=100, alpha=0.1):
        self.max_ite = max_ite  # maximum iterations
        self.env = GridWorld()  # assign the grid world as the environment
        self.alpha = alpha
        
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
            nx, ny = self.take_action(x, y)
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
                if state not in self.env.terminal:
                    x, y = state
                    nx,ny = episode[0]
                    # V(S) = V(S)+a*(R+y*V(S')-V(S))
                    self.env.values[x, y] = self.env.values[x, y] + self.alpha*(
                        self.env.reward + self.env.gamma*self.env.values[nx, ny]-self.env.values[x, y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ite', type=int, default=1000,
                        help="maximum iterations")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="step size")
    args = parser.parse_args()
    max_ite = args.ite
    alpha = args.alpha

    TDtest = TemporalDifference(max_ite=max_ite, alpha=alpha)
    TDtest.main_loop()
    print("after {} iterations".format(TDtest.max_ite))
    TDtest.env.printValues()
