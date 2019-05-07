ENV_FC = ['MountainCar-v0'] # envs only need full connected layer
LOWEST = -0.6
GOAL = 0.5
REWARD_SCALE = 10
GOAL_REWARD = 1000

class Config(object):
    display = True
    is_train = False
    enable_replay = True
    enable_target = False
    step_amount = 500
    store_after_steps = 100
    batch_size = 32
    discount = 0.99
    learning_rate = 0.001
    tagt_q_step = 100
    queue_size = 50
    epsilon = 1
    epsilon_decay = 0.05
    epsilon_min = 0.01
    max_step = 200
    test_times = 50
    rewards_file = 'training_rewards.json'
    env = 'MountainCar-v0' # fixed constant here

def config_init(FLAGS):
    configs = Config()
    for k, v in FLAGS.__flags.items():
        if hasattr(configs,k):
            setattr(configs,k,v.value)
    return configs
