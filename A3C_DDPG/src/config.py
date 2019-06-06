ENV_FC = ['Pendulum-v0'] # envs only need full connected layer

class Config(object):
    display = True
    is_train = False
    algorithm = "A3C"
    multi_agent = 3
    ep_amount = 2000
    store_after_eps = 100
    discount = 0.90
    actor_lr = 0.0001
    critic_lr = 0.001
    entropy = 0.01
    max_step = 200
    update_step = 10
    test_times = 50
    memory = 1000000
    noise = 0.1
    batch_size = 32
    queue_size = 10000
    l2 = 0
    rewards_file = 'training_rewards.json'
    algorithm = 'A3C'
    env = 'Pendulum-v0' # fixed constant here

def config_init(FLAGS):
    configs = Config()
    for k, v in FLAGS.__flags.items():
        if hasattr(configs,k):
            setattr(configs,k,v.value)
    return configs
