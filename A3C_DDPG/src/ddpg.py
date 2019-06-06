import inspect
import pprint
import os
import tensorflow as tf
from collections import deque
import time
import random
import numpy as np
import json
from .agent import Agent
from .ou_process import OrnsteinUhlenbeckActionNoise
import multiprocessing
import threading
import shutil

GLOBAL_SCOPE = 'Global'


def class_vars(obj):
	return {k: v for k, v in inspect.getmembers(obj)
			if not k.startswith('_') and not callable(k)}


class DDPGAgent(Agent):
	def __init__(self, config, env, sess, recorded_args=['ep_amount', 'entropy']):
		super(DDPGAgent, self).__init__(config, env, sess, recorded_args)
		# build network, loss and optimizer
		self.sess = sess
		self.actor_optimizer = tf.train.AdamOptimizer(
			self.config.actor_lr, name='AdamPropA')  # optimizer for the actor
		self.critic_optimizer = tf.train.AdamOptimizer(
			self.config.critic_lr, name='AdamPropC')  # optimizer for the critic
		self.build()
		# memory setting
		self.sample_deque = deque()
		# initialize session
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def build(self):
		with tf.variable_scope("train"):
			self.state = tf.placeholder(
				tf.float32, ((None,) + self.state_size), 'S')
			self.action = tf.placeholder(
				tf.float32, ((None,) + self.action_size), 'A'
			)
			self.grads = tf.placeholder(
				tf.float32, ((None,) + self.action_size), 'G'
			)
			self.target_Q = tf.placeholder(
					tf.float32, [None, 1], 'Q')

		self.a_out, self.c_out, self.a_params, self.c_params, self.action_grads = self.build_net(
			"local")  # get mu and sigma of estimated action from neural net
		self.target_a_out, self.target_c_out, self.target_a_params, self.target_c_params, _ = self.build_net("target")
		
		
		with tf.variable_scope("train"):
			with tf.variable_scope('actor'):
					grads = tf.gradients(
						self.a_out, self.a_params, -self.grads)
					grads_scaled = list(
						map(lambda x: tf.divide(x, self.config.batch_size), grads))
					# tf.gradients sums over the batch dimension here, must therefore divide by batch_size to get mean gradients
					self.a_train = self.actor_optimizer.apply_gradients(
						zip(grads_scaled, self.a_params))

			with tf.variable_scope('critic'):
					loss = tf.losses.mean_squared_error(
						self.target_Q, self.c_out)
					# add regularization term in case overfitting
					l2_loss = tf.add_n([tf.nn.l2_loss(
						v) for v in self.c_params if 'kernel' in v.name]) * self.config.l2
					total_loss = loss + l2_loss
					update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'critic')  # Ensure batch norm moving means and variances are updated every training step
					with tf.control_dependencies(update_ops):
						self.c_train = self.critic_optimizer.minimize(
							total_loss, var_list=self.c_params)
			
			with tf.name_scope('update'):
				self.update_target_op_a = [l_p.assign(t_p) for l_p, t_p in zip(
							self.a_params, self.target_a_params)]
				self.update_target_op_c = [l_p.assign(t_p) for l_p, t_p in zip(
							self.c_params, self.target_c_params)]

	def build_net(self, scope):
		with tf.variable_scope(scope):
			with tf.variable_scope('actor'):
				# approximate the action choose function
				a_dense1 = tf.layers.dense(
					self.state, 400, tf.nn.relu, name='dense1')
				a_dense2 = tf.layers.dense(
					a_dense1, 300, tf.nn.relu, name='dense2')
				a_out = tf.layers.dense(
					a_dense2, np.prod(self.action_size), tf.nn.tanh)
				a_out = tf.multiply(0.5, tf.multiply(
					a_out, (self.action_lh[1]-self.action_lh[0])) + (self.action_lh[0]+self.action_lh[1]))
			with tf.variable_scope('critic'):
				# approximate Q action-value function
				c_dense1 = tf.layers.dense(
					self.state, 400, tf.nn.relu, name='dense1')
				c_dense_s = tf.layers.dense(c_dense1, 300, name="dense_s")
				c_dense_a = tf.layers.dense(self.action, 300, name="dense_a")
				c_dense2 = tf.nn.relu(c_dense_a+c_dense_s)
				c_out = tf.layers.dense(c_dense2, 1, name="c_out")
			a_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
			c_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
			# gradient of value output wrt action input (update the actor)
			action_grads = tf.gradients(c_out, self.action)
		return a_out, c_out, a_params, c_params, action_grads

	def store_memory(self, state, action, reward, next_state, done):
		self.sample_deque.append(
			(state, action, reward, next_state, done))
		if len(self.sample_deque) > self.config.queue_size:
			self.sample_deque.popleft()
		if len(self.sample_deque) > self.config.batch_size:
			self.train_network()

	def train_network(self):
		minibatch = random.sample(
			self.sample_deque, self.config.batch_size)
		state_batch = [data[0] for data in minibatch]   # state
		action_batch = [data[1] for data in minibatch]  # action
		reward_batch = [data[2] for data in minibatch]  # reward
		next_state_batch = [data[3] for data in minibatch]  # next_state
		done_batch = [data[4] for data in minibatch]
		
		# Critic  
		next_action = self.sess.run(self.target_a_out, {self.state:next_state_batch})  
		next_Q = self.sess.run(self.target_c_out, {self.state:next_state_batch, self.action:next_action})[:,0]  
		next_Q[done_batch] = 0
		target_Q = reward_batch + (next_Q*self.config.discount)
		self.sess.run(self.c_train, {self.state:state_batch, self.action:action_batch, self.target_Q:np.expand_dims(target_Q, 1)})   
		
		# Actor
		actor_action = self.sess.run(self.a_out, {self.state:state_batch})
		action_grads = self.sess.run(self.action_grads, {self.state:state_batch, self.action:actor_action})
		self.sess.run(self.a_train, {self.state:state_batch, self.grads:action_grads[0]})
        
		# Update target networks
		self.update_target()


	def update_target(self):
		self.sess.run([self.update_target_op_a,self.update_target_op_c])

	def train(self):
		exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(np.prod(self.action_size)))
		self.update_target()
		step_num = 1
		avg_rewards = []
		for step_num in range(1,self.config.ep_amount+1):
			totle_reward = 0
			state = self.env.reset()
			exploration_noise.reset()
			for _ in range(self.config.max_step):
				self.env.render()
				action = self.sess.run(self.a_out,{self.state:state[np.newaxis,:]})[0]
				action += exploration_noise() * self.config.noise
				next_state, reward, done, _ = self.env.take_action(action)
				# train step is carried only when we have stored enough experiences
				self.store_memory(state, action, reward, next_state, done)
				state = next_state
				totle_reward += reward
				if done:
					break
			avg_reward = totle_reward/self.config.max_step
			print('train step:{}\tavg_reward:{}'.format(step_num, avg_reward))
			avg_rewards.append(avg_reward)
			if step_num % self.config.store_after_eps == 0:
				self.save_network(step_num)
		with open("DDPG_"+self.config.rewards_file,"w") as f:
			f.write(json.dumps(avg_rewards))


	def test(self):
		self.load_network()
		for j in range(self.config.test_times):
			totle_reward = 0
			state = self.env.reset()
			for i in range(1, self.config.max_step):
				self.env.render()
				action = self.sess.run(self.a_out,{self.state:np.reshape(state,(1,state.shape[0]))})
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
