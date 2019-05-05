import tensorflow as tf 
from config import config_init, ENV_FC
import random
from dqn.agent import Agent
from dqn.env import Environment

flags = tf.app.flags
# global configs
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_boolean('display', True, 'Whether to do display the env or not')
flags.DEFINE_boolean('is_train', False, 'Whether to train or test')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction')

# para configs
flags.DEFINE_integer('tagt_q_step', 100, 'step interval to update target q')
flags.DEFINE_float('epsilon', 0.01, 'probability to take random exploration')
flags.DEFINE_integer('queue_size', 500, 'idx / # of gpu fraction')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

# version configs
flags.DEFINE_boolean('enable_replay', True, 'Whether to use experience replay')
flags.DEFINE_boolean('enable_target', True, 'Whether to use target network')


FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

# calculate the fraction from str args
def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)
  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
    # set the percentage of GPU usage
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction) 
    )
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        configs = config_init(FLAGS)

        # add other environments in the future
        if configs.env not in ENV_FC:
            raise ValueError("can't use environment except MountainCar-v0")
        env = Environment(configs)
        
        agent = Agent(configs,env,sess)
        
        if configs.is_train:
            agent.train()
        else:
            agent.test()
    
if __name__ == "__main__":
    tf.app.run()
