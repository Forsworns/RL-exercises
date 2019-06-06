import inspect
import pprint
import os
import tensorflow as tf
import time
import random
import numpy as np
import json
from .agent import Agent
import multiprocessing
import threading

GLOBAL_SCOPE = 'Global'


def class_vars(obj):
	return {k: v for k, v in inspect.getmembers(obj)
			if not k.startswith('_') and not callable(k)}


class A3CAgent(Agent):
	def __init__(self, config, env, sess, scope, recorded_args=['ep_amount', 'entropy'], globalNet=None):
		super(A3CAgent, self).__init__(config, env, sess, recorded_args)
		# build network, loss and optimizer
		self.scope = scope
		self.globalNet = globalNet
		self.actor_optimizer = tf.train.RMSPropOptimizer(
			self.config.actor_lr, name='RMSPropA')  # optimizer for the actor
		self.critic_optimizer = tf.train.RMSPropOptimizer(
			self.config.critic_lr, name='RMSPropC')  # optimizer for the critic
		self.build(scope, globalNet)

		# initialize session
		self.sess.run(tf.global_variables_initializer())

	def build(self, scope, globalNet):
		if scope == GLOBAL_SCOPE:   # get global network
			with tf.variable_scope(scope):
				# get parameters of actor and critic net
				self.state = tf.placeholder(
					tf.float32, ((None,) + self.state_size), 'S')
				self.a_params, self.c_params = self.build_net(
					scope)[-2:]  
		else:   
			with tf.variable_scope(scope):
				# local net, calculate losses
				self.state = tf.placeholder(
					tf.float32, ((None,) + self.state_size), 'S')           
				self.action = tf.placeholder(
					tf.float32, ((None,) + self.action_size), 'A')        
				self.v_target = tf.placeholder(
					tf.float32, [None, 1], 'V')  # v_target value

				mu, sigma, self.v, self.a_params, self.c_params = self.build_net(
					scope)  # get mu and sigma of estimated action from neural net

				td = tf.subtract(self.v_target, self.v, name='TD_error')
				with tf.name_scope('c_loss'):
					self.c_loss = tf.reduce_mean(tf.square(td))

				with tf.name_scope('wrap_a_out'):
					mu, sigma = mu * self.action_lh[1], sigma + 1e-4

				normal_dist = tf.distributions.Normal(mu, sigma)

				with tf.name_scope('a_loss'):
					# read the paper, learn the reason for entropy loss term 
					log_prob = normal_dist.log_prob(self.action)
					exp_v = log_prob * td
					entropy = normal_dist.entropy()  # encourage exploration
					self.exp_v = self.config.entropy * entropy + exp_v
					self.a_loss = tf.reduce_mean(-self.exp_v)

				with tf.name_scope('choose_a'):  # use local params to choose action
					self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(
						1), axis=0), self.action_lh[0], self.action_lh[1])  # sample an action from distribution
				with tf.name_scope('local_grad'):
					# calculate gradients for the network weights
					self.a_train = tf.gradients(self.a_loss, self.a_params)
					self.c_train = tf.gradients(self.c_loss, self.c_params)

			with tf.name_scope('sync'):  # update local and global network weights
				with tf.name_scope('pull'):
					self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
						self.a_params, globalNet.a_params)]
					self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
						self.c_params, globalNet.c_params)]
				with tf.name_scope('push'):
					# use the state value function to approximate the Q (action-value) function
					self.update_a_op = self.actor_optimizer.apply_gradients(
						zip(self.a_train, globalNet.a_params))
					self.update_c_op = self.critic_optimizer.apply_gradients(
						zip(self.c_train, globalNet.c_params))

	def build_net(self, scope):
		# build network for actor and critics in different local threads
		w_init = tf.random_normal_initializer(0., .1)
		with tf.variable_scope('actor'):
			l_a = tf.layers.dense(self.state, 200, tf.nn.relu6,
								  kernel_initializer=w_init, name='la')
			# estimated action value
			mu = tf.layers.dense(l_a, np.prod(self.action_size), tf.nn.tanh,
								 kernel_initializer=w_init, name='mu')
			sigma = tf.layers.dense(
				l_a, np.prod(self.action_size), tf.nn.softplus, kernel_initializer=w_init, name='sigma')
		with tf.variable_scope('critic'):
			l_c = tf.layers.dense(self.state, 100, tf.nn.relu6,
								  kernel_initializer=w_init, name='lc')
			# estimated value for state
			v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
		a_params = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
		c_params = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
		return mu, sigma, v, a_params, c_params

	def update_global(self, feed_dict):
		self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

	def pull_global(self):
		self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

	def choose_action(self, state):
		state = state[np.newaxis, :]
		return self.sess.run(self.A, {self.state: state})[0]

	def train(self):
		global global_rewards, global_episodes, coord
		total_step = 1
		buffer_s, buffer_a, buffer_r = [], [], []
		while not coord.should_stop() and global_episodes < self.config.ep_amount:
			state = self.env.reset()
			ep_r = 0
			for ep_t in range(self.config.max_step):
				if self.scope == 'W_0' and self.config.display:
					self.env.render()
				# estimate stochastic action based on policy
				action = self.choose_action(state)
				next_s, r, done, _ = self.env.take_action(
					action)  # make step in environment
				done = True if ep_t == self.config.max_step - 1 else False

				ep_r += r
				# save actions, states and rewards in buffer (buffer size = how many steps to update global network)
				buffer_s.append(state)
				buffer_a.append(action)
				buffer_r.append((r+8)/8)    # normalize reward

				if total_step % self.config.update_step == 0 or done:   # update global and assign to local net
					if done:
						next_v = 0   # terminal
					else:
						next_v = self.sess.run(
							self.v, {self.state: next_s[np.newaxis, :]})[0, 0]
					buffer_v_target = []
					for r in buffer_r[::-1]:    # reverse buffer r
						next_v = r + self.config.discount * next_v
						buffer_v_target.append(next_v)
					buffer_v_target.reverse()

					buffer_s, buffer_a, buffer_v_target = np.vstack(
						buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
					feed_dict = {
						self.state: buffer_s,
						self.action: buffer_a,
						self.v_target: buffer_v_target,
					}
					# actual training step, update global ACNet
					self.update_global(feed_dict)
					buffer_s, buffer_a, buffer_r = [], [], []
					self.pull_global()  # get global parameters to local ACNet

				state = next_s
				total_step += 1
				if done:
					if len(global_rewards) < 5:  # record running episode reward
						global_rewards.append(ep_r)
					else:
						global_rewards.append(ep_r)
						# smoothing
						global_rewards[-1] = (np.mean(global_rewards[-5:]))
					print(
						self.scope,
						" ep:", global_episodes,
						" reward:{}".format(global_rewards[-1]),
					)
					global_episodes += 1
					break
			if global_episodes % self.config.store_after_eps == 0:
				self.save_network(global_episodes, [*self.globalNet.a_params,*self.globalNet.c_params])
								  # "actor": self.globalNet.a_params, "critic": self.globalNet.c_params})

	def test(self):
		self.saver = tf.train.Saver([*self.globalNet.a_params,*self.globalNet.c_params],max_to_keep=10)
		self.load_network()
		self.pull_global()
		for j in range(self.config.test_times):
			totle_reward = 0
			state = self.env.reset()
			for i in range(1, self.config.max_step):
				self.env.render()
				action = self.choose_action(state)
				next_state, reward, done, _ = self.env.take_action(action)
				state = next_state
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


def trainA3C(config, env, sess, recorded_args=['ep_amount', 'entropy']):
	global global_rewards, global_episodes, coord
	global_rewards = []
	global_episodes = 0
	with tf.device("/cpu:0"):
		global_ac = A3CAgent(
			config, env, sess, GLOBAL_SCOPE, recorded_args)
		workers = []
		for i in range(config.multi_agent):
			scope = 'W_%i' % i
			workers.append(A3CAgent(config, env, sess,
									scope, recorded_args, global_ac))

	coord = tf.train.Coordinator()
	sess.run(tf.global_variables_initializer())

	worker_threads = []
	for worker in workers:  # start threads
		def job(): return worker.train()
		t = threading.Thread(target=job)
		t.start()
		worker_threads.append(t)
	coord.join(worker_threads)  # wait for termination of threads
	with open("A3C_"+config.rewards_file, "w") as f:
		f.write(json.dumps(global_rewards))


def testA3C(config, env, sess, recorded_args=['algorithm','ep_amount', 'entropy']):
	global_ac = A3CAgent(config, env, sess, GLOBAL_SCOPE, recorded_args)
	agent = A3CAgent(config, env, sess, "test", recorded_args, globalNet=global_ac)
	agent.test()