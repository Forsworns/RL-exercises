ENV_FC = ['MountainCar-v0'] # envs only need full connected layer


class Config(object):
    display = True
    is_train = False
    enable_replay = True
    enable_target = True
    step_amount = 10000
    store_after_steps = 5000
    batch_size = 32
    discount = 0.99
    learning_rate = 0.0001
    tagt_q_step = 100
    queue_size = 500
    epsilon = 0.01
    max_step = 200
    test_times = 50
    rewards_file = 'training_rewards.json'
    losses_file = 'training_losses.json'
    env = 'MountainCar-v0' # fixed constant here

def config_init(FLAGS):
    configs = Config()
    for k, v in FLAGS.__flags.items():
        if hasattr(configs,k):
            setattr(configs,k,v.value)
    return configs
