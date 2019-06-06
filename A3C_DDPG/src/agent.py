import multiprocessing as mp
import inspect
import pprint
import os
import tensorflow as tf
import time
import random
import numpy as np
import json
from abc import ABCMeta, abstractmethod


def class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('_') and not callable(k)}


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, env, sess, recorded_args=['ep_amount']):
        self.sess = sess
        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.action_lh = env.action_lh

        # only recorded_args are used to name directories
        self.recorded_args = recorded_args
        self._attrs = class_vars(config)
        self.config = config
        pprint.PrettyPrinter().pprint(class_vars(config))

        # initialize session
        self.sess = sess

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @property
    def model_dir(self):
        model_dir = self.config.env
        for k, v in self._attrs.items():
            if k in self.recorded_args:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
                                            if type(v) == list else v)
        return model_dir + '/'

    def load_network(self):    
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Load failed")

    def save_network(self, step, params=None):
        if params != None:
            self.saver = tf.train.Saver(params,max_to_keep=10)
        else:
            self.saver = tf.train.Saver()
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver.save(
            self.sess, os.path.join(self.model_dir, 'Pendulum-{}'.format(step)))
        print('Save the model', 'Pendulum-{}'.format(step))
