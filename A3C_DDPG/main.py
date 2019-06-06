import tensorflow as tf
import multiprocessing as mp
from src.config import config_init, ENV_FC
import random
from src.a3c import trainA3C, testA3C
from src.ddpg import DDPGAgent
from src.env import Environment

flags = tf.app.flags
# global configs
flags.DEFINE_string('algorithm', 'DDPG', 'A3C or DDPG')
flags.DEFINE_boolean('display', True, 'Whether to do display the env or not')
flags.DEFINE_boolean('is_train', True, 'Whether to train or test')

# para configs
flags.DEFINE_integer('memory',100000,'the memory size')
flags.DEFINE_integer('ep_amount', 2000, 'maximum episode amount')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_integer('multi_agent', 4, 'how many agents are paralleled')
flags.DEFINE_float('noise',0.5,'the noise range over action space range')
flags.DEFINE_float('l2',0,'l2 regularization term weight')

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
