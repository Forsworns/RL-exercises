import tensorflow as tf
import multiprocessing as mp
from src.config import config_init, ENV_FC
import random
from src.a3c import trainA3C, testA3C
from src.ddpg import DDPGAgent
from src.env import Environment

flags = tf.app.flags
# global configs
flags.DEFINE_string('algorithm', 'A3C', 'A3C or DDPG')
flags.DEFINE_boolean('display', True, 'Whether to do display the env or not')
flags.DEFINE_boolean('is_train', True, 'Whether to train or test')

# para configs
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_integer('ep_amount', 2000, 'Maximum episode amount')
flags.DEFINE_integer('store_after_eps', 100, 'How many episodes to store')
flags.DEFINE_integer('max_step', 200, 'Max steps in an episode')
flags.DEFINE_integer('test_times', 50, 'Experiment times when testing')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_float('discount', 0.99, 'The discount gamma for future reward')
flags.DEFINE_float('actor_lr', 0.0001, 'Learning rate for actor')
flags.DEFINE_float('critic_lr', 0.001, 'Learning rate for critic')

# A3C config
flags.DEFINE_float('entropy', 0.01, 'Entropy loss term weight')
flags.DEFINE_integer('multi_agent', 4, 'How many agents are paralleled')
flags.DEFINE_integer('update_step', 10, 'How many steps to update the global network')

# DDPG config
flags.DEFINE_integer('memory',100000,'The memory size')
flags.DEFINE_float('noise',0.5,'The noise range over action space range')
flags.DEFINE_float('l2',0,'L2 regularization term weight')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.multi_agent > mp.cpu_count():
    print("can't assign agents more than {} cpu core".format(mp.cpu_count()))


def main(_):
    # set the percentage of GPU usage
    FLAGS.multi_agent = min(FLAGS.multi_agent, mp.cpu_count())
    with tf.Session(config=tf.ConfigProto()) as sess:
        configs = config_init(FLAGS)
        # add other environments in the future
        if configs.env not in ENV_FC:
            raise ValueError("can't use environment except Pendulum-v0")
        env = Environment(configs)

        if configs.algorithm == "A3C":
            if configs.is_train:
                trainA3C(configs, env, sess, ['algorithm','ep_amount', 'entropy'])
            else:
                testA3C(configs, env, sess, ['algorithm','ep_amount', 'entropy'])
        elif configs.algorithm == "DDPG":
            agent = DDPGAgent(configs, env, sess, recorded_args=['algorithm','ep_amount', 'entropy'])
            if configs.is_train:
                agent.train()
            else:
                agent.test()
        else:
            print("choose A3C or DDPG algorithm")


if __name__ == "__main__":
    tf.app.run()
