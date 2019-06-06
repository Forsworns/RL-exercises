import gym

class Environment(object):
    def __init__(self,config):
        self.env = gym.make(config.env)
        if hasattr(self.env,'unwrapped'):
            self.env = self.env.unwrapped
        self.display = config.display
    
    def reset(self):
        return self.env.reset()

    @property
    def action_size(self):
        return self.env.action_space.shape

    @property
    def action_lh(self):
        return [self.env.action_space.low, self.env.action_space.high]

    @property
    def state_size(self):
        return self.env.observation_space.shape

    # for compare
    def random_action(self):
        action = self.env.action_space.sample()
        return self.take_action(action)

    def take_action(self,action):
        return self.env.step(action)

    def render(self):
        if self.display:
            self.env.render()