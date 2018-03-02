import numpy as np
import random


#### alpha_theta_mu = 10.0 * alpha_theta_sigma

# make a sigma max (eg max velocity) and a sigma min

class DiscreteActorCritic:
	def __init__(self, numFeatures, numActions, alpha, lamda, gamma, name, id=0, w_lambda=None):
		self.name = name

		self.numFeatures = numFeatures
		self.numActions = numActions
		self.lamda = lamda
		self.w_lamda = w_lambda if w_lambda != None else lamda
		self.gamma = gamma

		self.alpha = alpha
		self.alpha_r = alpha*.05
		self.alpha_w = alpha
		self.alpha_theta = alpha*0.1

		self.reset()

	def reset(self):
		# critic
		self.w = np.zeros(self.numFeatures)
		self.w_traces = np.zeros(self.numFeatures)

		# actor
		self.theta = np.zeros((self.numFeatures, self.numActions))
		self.theta_traces = np.zeros((self.numFeatures, self.numActions))
		self.pi = np.zeros(self.numActions)

		self.average_r = 0.0

	def cut_traces(self):
		self.w_traces = np.zeros(self.numFeatures)
		self.theta_traces = np.zeros((self.numFeatures, self.numActions))
		self.pi = np.zeros(self.numActions)

	def update(self, action, phi_t, phi_t_1, r):

		delta = r - self.average_r + self.gamma*np.dot(self.w, phi_t_1) - np.dot(self.w, phi_t)

		self.average_r += self.alpha_r*delta

		# critic
		self.w_traces *= self.lamda*self.gamma
		self.w_traces += phi_t
		self.w += self.alpha_w*delta*self.w_traces

		# calc compatable features
		piSum = np.zeros((self.numFeatures, self.numActions))
		for a_i in range(self.numActions):
			Phi = np.zeros((self.numFeatures, self.numActions))
			Phi[:, a_i] = phi_t
			piSum += self.pi[a_i]*Phi

		Phi = np.zeros((self.numFeatures, self.numActions))
		Phi[:, action] = phi_t



		compatibleFeatures = Phi - piSum

		# actor
		self.theta_traces *= self.lamda
		self.theta_traces += compatibleFeatures

		self.theta += self.alpha_theta*delta*self.theta_traces

		return delta

	# softmax action
	def getAction(self, phi):
		phi = phi.reshape(self.numFeatures, 1)
		self.pi = np.exp(np.dot(self.theta.T,phi))
		self.pi /= sum(self.pi)

		prob = random.random()
		psum = 0.0
		for a_i in range(self.numActions):
			if prob < (self.pi[a_i] + psum):
				return a_i, self.pi
			else:
				psum += self.pi[a_i]

	def save(self):
		np.savez(self.name+'_weights', \
		 w=self.w, \
		 w_traces=self.w_traces, \
		 theta=self.theta, \
		 theta_traces=self.theta_traces, \
		 average_r=self.average_r)

	def load(self):
		data = np.load(self.name+'_weights.npz')

		# critic
		self.w = data['w']
		self.w_traces = data['w_traces']

		# actor
		self.theta = data['theta']
		self.theta_traces = data['theta_traces']
		self.pi = np.zeros(self.numActions)

		self.average_r = data['average_r']

class ContinuousActorCritic:
	def __init__(self, numFeatures, alpha, lamda, gamma, name='', id=0, max_sigma = 10, min_sigma=0.001, max_mean=None, min_mean=None, w_lamda=None):
		self.name = name

		self.numFeatures = numFeatures
		self.lamda = lamda
		self.w_lamda = w_lamda if w_lamda != None else lamda

		self.alpha = alpha
		self.alpha_r = alpha*.001
		self.alpha_theta = np.concatenate((np.array([.5*alpha]*numFeatures), np.array([.1*alpha]*numFeatures)), axis=0)
		self.alpha_v = alpha

		self.gamma = gamma


		self.max_sigma = max_sigma
		self.min_sigma = min_sigma

		self.max_mean = max_mean
		self.min_mean = min_mean

		self.reset()

	def newEpisode(self):
		self.critic_value = .0
		self.v_traces = np.zeros(self.numFeatures)
		self.theta_traces = np.zeros(2*self.numFeatures)
		self.average_r = 0.0

	def reset(self):
		# critic

		self.critic_value = 0.0

		self.v = np.zeros(self.numFeatures)
		self.w = np.zeros(2*self.numFeatures)

		self.theta = np.zeros(2*self.numFeatures)

		self.mean = 0
		self.sigma = 1

		self.newEpisode()

	def getValue(self, phi):
		return np.dot(self.v, phi)

	def update(self, action, phi_t, phi_t_1, r):

		self.critic_value = np.dot(self.v, phi_t_1)

		delta = r - self.average_r + self.gamma*np.dot(self.v, phi_t_1) - np.dot(self.v, phi_t)

		self.average_r += self.alpha_r*delta


		self.v_traces *= self.gamma*self.lamda
		self.v_traces += phi_t

		self.v += self.alpha_v*delta*self.v_traces

		theta_grad_mu = (action-self.mean)*phi_t
		theta_grad_sigma = ((action - self.mean)**2 - self.sigma**2)*phi_t
		theta_grad = np.concatenate((theta_grad_mu, theta_grad_sigma), axis=0)

		assert(len(theta_grad) == 2*self.numFeatures)

		self.theta_traces *= self.lamda*self.gamma
		self.theta_traces += theta_grad

		#incremental nat actor critic
		#self.w = self.w - self.alpha_v*theta_grad*(theta_grad.T*self.w) + self.alpha_v*delta*self.theta_traces

		self.theta += self.alpha_theta*delta*self.theta_traces #np.multiply(self.alpha_theta, self.w)

		self.mean = np.dot(self.theta[:self.numFeatures],phi_t_1)
		self.sigma = np.exp(np.dot(self.theta[self.numFeatures:],phi_t_1))

		return delta

	def getAction(self, phi):
		self.mean = np.dot(self.theta[:self.numFeatures],phi)
		self.sigma = np.exp(np.dot(self.theta[self.numFeatures:],phi))
		if not self.max_sigma == None and self.sigma > self.max_sigma:
			self.sigma = self.max_sigma
		elif self.sigma < self.min_sigma:
			self.sigma = self.min_sigma

		if not self.max_mean == None and self.mean > self.max_mean:
			self.mean = self.max_mean
		elif not self.min_mean == None and self.mean < self.min_mean:
			self.mean = self.min_mean

		action = np.random.normal(self.mean, self.sigma)
		return action

	def save(self):
		np.savez(self.name+'_weights', \
			v=self.v, \
			v_traces=self.v_traces, \
			w=self.w, \
			theta=self.theta, \
			theta_traces=self.theta_traces, \
			average_r=self.average_r)

	def load(self):
		data = np.load(self.name+'_weights.npz')

		# critic
		self.v = data['v']
		self.v_traces = data['v_traces']
		self.w = data['w']

		# actor
		self.theta = data['theta']
		self.theta_traces = data['theta_traces']

		self.average_r = data['average_r']

class MultivariateContinuousActorCritic:
	def __init__(self, numFeatures, actionDim, alpha, lamda, gamma, name, id=0):
		self.numFeatures = numFeatures
		self.actionDim = actionDim
		self.lamda = lamda
		self.alpha = alpha
		self.alpha_r = alpha*.001
		self.alpha_v = alpha # critic

		self.alpha_theta = np.concatenate((np.array([10*alpha]*numFeatures), np.array([alpha]*numFeatures)), axis=0) # policy

		self.gamma = gamma
		self.reset()

	def reset(self):
		self.v_traces = np.zeros(numFeatures)
		self.v = np.zeros(numFeatures)

		self.w = np.zeros((2*numFeatures, actionDim))

		self.theta_traces = np.zeros((2*numFeatures, actionDim))
		self.theta = np.random.random((2*numFeatures, actionDim))

		self.theta_grad = np.random.random((2*numFeatures, actionDim))

		self.mu = np.zeros(actionDim)
		self.sigma = np.random.random((actionDim, actionDim))

		self.average_r = 0.0

	def update(self, action, phi_t, phi_t_1, r):

		delta = r - self.average_r + self.gamma*np.dot(self.v, phi_t_1) - np.dot(self.v, phi_t)

		self.average_r += self.alpha_r*delta

		self.v_traces *= self.gamma*self.lamda
		self.v_traces += phi_t

		self.v += self.alpha_v*delta*self.v_traces

		try:
			invSigma = np.linalg.inv(self.sigma)
		except Exception as e:
			print(e)
			print(self.sigma)

		actDiff = (action - self.mean)

		theta_grad_mu = invSigma*actDiff*phi_t
		theta_grad_sigma = -.5*(invSigma - invSigma*actDiff*actDiff.T*invSigma)*np.tile(phi_t, (self.actionDim, 1)) # repeat features

		theta_grad = np.concatenate((theta_traces_mu, theta_grad_sigma), axis=1)
		assert(theta_grad.shape == (self.actionDim, 2*self.numFeatures))

		self.theta_traces *= self.gamma*self.lamda
		self.theta_traces += theta_grad

		self.w = self.w - self.alpha_v*self.theta_grad*self.theta_grad.T*self.w + self.alpha_theta*delta*self.theta_traces

		self.theta += self.alpha_theta*self.w

	def getAction(self, phi):
		self.mean = np.dot(self.theta[:self.numFeatures].T,phi)
		print('Mean vector: ' + str(self.mean) + ' shape ' + str(self.mean.shape))


		self.sigma = np.exp(np.dot(self.theta[self.numFeatures:].T, np.tile(phi, (self.actionDim, 1)).T))

		print('Sigma matrix: ' + str(self.sigma) + ' shape ' + str(self.sigma.shape))

		tempSigma = self.sigma.T[0]

		print('tempSigma: ' + str(tempSigma))

		self.sigma = np.fill_diagonal(tempSigma)

		print('Sigma: ' + str(self.sigma))

		# if not np.all(np.linalg.eigvals(self.sigma) > 0): # check that sigma is pos def
		# 	self.sigma =

		action = np.random.multivariate_normal(self.mean, self.sigma)
		assert(len(action) == self.actionDim)
		return action

	def save(self):
		np.savez(self.name+'_weights', \
			v=self.v, \
			v_traces=self.v_traces, \
			w=self.w, \
			theta=self.theta, \
			theta_traces=self.theta_traces, \
			theta_grad=self.theta_grad, \
			average_r=self.average_r)

	def load(self):
		data = np.load(self.name+'_weights.npz')

		# critic
		self.v = data['v']
		self.v_traces = data['v_traces']
		self.w = data['w']

		# actor
		self.theta = data['theta']
		self.theta_traces = data['theta_traces']
		self.theta_grad = data['theta_grad']

		self.average_r = data['average_r']
