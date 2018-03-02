import numpy as np
import math
import random
from horde import *

class Policy:

	def __init__(self, numFeatures=10, numActions=1, epsilon=0, id=0, w=[], name=''):
		self.numFeatures = numFeatures if len(w) == 0 else w.shape[0]
		self.numActions = numActions if len(w) == 0 else w.shape[1]
		self.epsilon = epsilon
		self.w = np.zeros((numFeatures, numActions)) if len(w) == 0 else w
		self.id = id
		self.name = name

	def setWeights(self, _w):
		try:
			assert(self.w.shape == _w.shape)
			self.w = _w
		except:
			raise Exception('Make sure the weights set for policy ' + str(self.id) + ' have the shape ' + str(self.w.shape) + '. Got ' + str(_w.shape) + '.')

	def getEpsilonAction(self, phi):

		if np.random.random() < self.epsilon:
			self.actionProbs = 1/self.numActions
			return np.random.randint(0,self.numActions)
		return self.getAction(phi)

	def getAction(self, phi):
		a_vals = sum(np.compress(phi, self.w, axis=0))
		action = np.argwhere(a_vals == np.amax(a_vals)).flatten().tolist()
		action = random.choice(action)
		self.actionProbs = float(a_vals[action]/sum(a_vals))
		return action

	def getActionProb(self, action, phi):
		a_vals = sum(np.compress(phi, self.w, axis=0)) #http://ipython-books.github.io/featured-01/
		return a_vals[action]/sum(a_vals)

class PolicyLegion:
	def __init__(self):
		self.policies = []
		self.behaviourIndex = 0
		self.initialBehaviorIndex = 0
		self.Rho = []
		self.reflexes = []

	def setBehaviour(self, id):
		if id != self.behaviourIndex:
			self.behaviourIndex = id
			print('Changed Legion Policy to ' + str(self.policies[self.behaviourIndex].name))

	def addPolicy(self, policy):
		policy.id = len(self.policies)
		self.policies.append(policy)
		self.Rho.append(0)

	def addReflex(self, reflex):
		self.reflexes.append(reflex)

	def updateReflex(self, predictions):
		activations = [ref.update(predictions[ref.prediction_id]) for ref in self.reflexes]
		maxActivation = max(activations)

		if maxActivation[0] > 0:
			if maxActivation[1] == self.behaviourIndex:
				return
			self.setBehaviour(maxActivation[1])

		else:
			self.setBehaviour(self.initialBehaviorIndex)


	def getAction(self, phi):
		maxAction = self.policies[self.behaviourIndex].getAction(phi)
		self.Rho = [p.getActionProb(maxAction, phi)/self.policies[self.behaviourIndex].actionProbs for p in self.policies]
		self.Rho /= sum(self.Rho)
		return maxAction

	def getActionProbs(self, action, phi):
		mu_p = self.policies[self.behaviourIndex].getActionProb(action, phi)
		mu_p = mu_p + 0.001 if mu_p == 0 else mu_p

		Rho = [p.getActionProb(action, phi)/mu_p for p in self.policies]
		Rho /= sum(Rho)
		return Rho

class PolicyGenerator:

	def __init__(self, numFeatures=10, numActions=1, alpha = .01):
		self.numFeatures = numFeatures
		self.numActions = numActions
		self.w = np.zeros((numFeatures, numActions))
		self.alpha = alpha

	def update(self, phi, action):
		inital = sum(self.w)

		self.w[:,action] += phi*self.alpha

		for a in range(self.numActions):
			try:
				self.w[:, a] /= sum(self.w[:, a])
			except:
				pass
		return sum(inital - sum(self.w))

class PolicyFactory:

	def __init(self):
		self.policyGenerators = []

	def addGenerator(self, generator):
		self.policyGenerators.append(generator)

	def update(self, phi, action):
		for generator in self.policyGenerators:
			generator.update(phi, action)


class Reflex:

	def __init__(self, policy_id, prediction_id, threshold):
		self.policy_id = policy_id
		self.prediction_id = prediction_id
		self.threshold = threshold
		self.activation = 0

	def update(self, prediction):
		if prediction > self.threshold:
			self.activation = prediction - self.threshold
		else:
			self.activation = 0

		return (self.activation, self.policy_id)
