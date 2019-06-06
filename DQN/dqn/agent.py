import inspect
import pprint
import os
import tensorflow as tf
from collections import deque
import time
import random
import numpy as np
import json
from config import LOWEST, GOAL, GOAL_REWARD, REWARD_SCALE


def class_vars(obj):
	return {k: v for k, v in inspect.getmembers(obj)
			if not k.startswith('_') and not callable(k)}


class Agent(object):
	def __init__(self, config, env, sess, recorded_args=['enable_target', 'tagt_q_step', 'queue_size']):
		self._saver = None
		self.env = env
		self.state_size = env.state_size
		self.action_size = env.action_size

		# only recorded_args are used to name directories
		self.recorded_args = recorded_args
		self._attrs = class_vars(config)
		self.config = config
		pprint.PrettyPrinter().pprint(class_vars(config))

		# memory setting
		self.sample_deque = deque()

		# build network, loss and optimizer
		self.sess = sess
		self.full_connected()
		self.loss_optimizer()

		# initialize session
		self.sess.run(tf.global_variables_initializer())
		self.load_network()

	def full_connected(self):
		# envs like MountainCar-v0 only need full connected
		self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
		# realtime Q-network
		with tf.variable_scope('realtime_net'):
			W1 = self.weight_variable([self.state_size, 20])
			b1 = self.bias_variable([20])
			W2 = self.weight_variable([20, 48])
			b2 = self.bias_variable([48])
			W3 = self.weight_variable([48, self.action_size])
			b3 = self.bias_variable([self.action_size])
			layer1_out = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
			layer2_out = tf.nn.relu(tf.matmul(layer1_out, W2) + b2)
			self.network_out = tf.matmul(layer2_out, W3) + b3
		with tf.variable_scope('target_net'):
			# target Q-network
			t_W1 = self.weight_variable([self.state_size, 20])
			t_b1 = self.bias_variable([20])
			t_W2 = self.weight_variable([20, 48])
			t_b2 = self.bias_variable([48])
			t_W3 = self.weight_variable([48, self.action_size])
			t_b3 = self.bias_variable([self.action_size])
			t_layer1_out = tf.nn.relu(tf.matmul(self.state_input, t_W1) + t_b1)
			t_layer2_out = tf.nn.relu(tf.matmul(t_layer1_out, t_W2) + t_b2)
			self.t_network_out = tf.matmul(t_layer2_out, t_W3) + t_b3

		r_params = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES, scope='realtime_net')
		t_params = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		with tf.variable_scope('update_target'):
			self.target_update_op = [
				tf.assign(t, r) for t, r in zip(t_params, r_params)]

	def loss_optimizer(self):
		self.action_input = tf.placeholder(
			tf.float32, [None, self.action_size])
		self.y_input = tf.placeholder(tf.float32, [None])
		q_reward = tf.reduce_sum(tf.multiply(
			self.network_out, self.action_input), reduction_indices=1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - q_reward))
		self.optimizer = tf.train.AdamOptimizer(
			self.config.learning_rate).minimize(self.cost)

	def store_memory(self, state, action, reward, next_state, done):
		one_hot_action = np.zeros(self.action_size)
		one_hot_action[action] = 1
		self.sample_deque.append(
			(state, one_hot_action, reward, next_state, done))
		if self.config.enable_replay:
			if len(self.sample_deque) > self.config.queue_size:
				self.sample_deque.popleft()
			if len(self.sample_deque) > self.config.batch_size:
				self.train_network()
		else:
			self.train_network()

	def train_network(self):
		if self.config.enable_replay:
			minibatch = random.sample(
				self.sample_deque, self.config.batch_size)
		else:
			minibatch = self.sample_deque.pop()  # only one sample, in fact
		state_batch = [data[0] for data in minibatch]   # state
		action_batch = [data[1] for data in minibatch]  # action
		reward_batch = [data[2] for data in minibatch]  # reward
		next_state_batch = [data[3] for data in minibatch]  # next_state
		y_batch = []
		if self.config.enable_target:
			q_value_batch = self.t_network_out.eval(
				feed_dict={self.state_input: next_state_batch})
		else:
			q_value_batch = self.network_out.eval(
				feed_dict={self.state_input: next_state_batch})
		for i in range(len(minibatch)):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] +
							   self.config.discount * np.max(q_value_batch[i]))
		self.optimizer.run(feed_dict={
			self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch})

	def weight_variable(self, shape):
		init = tf.truncated_normal(shape)
		return tf.Variable(init)

	def bias_variable(self, shape):
		init = tf.truncated_normal(shape)
		return tf.Variable(init)

	def greedy_action(self, state):
		return np.argmax(self.network_out.eval(feed_dict={self.state_input: [state]})[0])

	def choose_action(self, state):
		self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon)
		# epsilon-greedy algorithm (whether explore or simply greedy choose actions)
		if np.random.rand() <= self.config.epsilon:
			return np.random.randint(0, self.action_size - 1)
		else:
			return self.greedy_action(state)

	def load_network(self):
		checkpoint = tf.train.get_checkpoint_state(self.model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Load failed")

	def save_network(self, step):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		save_path = self.saver.save(
			self.sess, os.path.join(self.model_dir, 'mountain_car-{}'.format(step)))
		print('Save the model','mountain_car-{}'.format(step))

	def train(self):
		step_num = 0
		avg_rewards = []
		for step_num in range(1,self.config.step_amount+1):
			totle_reward = 0
			state = self.env.reset()
			for _ in range(self.config.max_step):
				self.env.render()
				action = self.choose_action(state)
				next_state, reward, done, _ = self.env.take_action(action)
				position, _ = next_state
				if position >= GOAL:
					reward += GOAL_REWARD
				self.store_memory(state, action, reward, next_state, done)
				state = next_state
				totle_reward += reward
				if done:
					break
			if self.config.epsilon > self.config.epsilon_min:
				self.config.epsilon -= self.config.epsilon_decay # each time reduce the searching probability
			if step_num % self.config.tagt_q_step == 0:
				self.sess.run(self.target_update_op)
			avg_reward = totle_reward/self.config.max_step
			print('train step:{}\tavg_reward:{}'.format(step_num, avg_reward))
			avg_rewards.append(avg_reward)
			if step_num % self.config.store_after_steps == 0:
				self.save_network(step_num)
		with open(self.config.rewards_file,"w") as f:
			f.write(json.dumps(avg_rewards))

	def test(self):
		for j in range(self.config.test_times):
			totle_reward = 0
			state = self.env.reset()
			for i in range(1, self.config.max_step):
				self.env.render()
				action = self.greedy_action(state)
				next_state, _, done, _ = self.env.take_action(action)
				state = next_state
				position, _ = next_state
				reward = abs(position-LOWEST)*REWARD_SCALE
				if position >= GOAL:
					reward += GOAL_REWARD
				totle_reward += reward
				if done:
					print('step:', i)
					if i == self.config.max_step:
						print('failed!')
						break
					else:
						print("success!")
						time.sleep(3)
						break
			avg_reward = totle_reward / self.config.max_step
			print('avg_reward:{}'.format(avg_reward))

	@property
	def model_dir(self):
		model_dir = self.config.env
		for k, v in self._attrs.items():
			if k in self.recorded_args:
				model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v])
										 if type(v) == list else v)
		return model_dir + '/'

	@property
	def saver(self):
		if self._saver is None:
			self._saver = tf.train.Saver(max_to_keep=10)
		return self._saver
