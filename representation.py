import numpy as np


class SelectiveKanervaCoder:

	def __init__(self, _numPrototypes, _dimensions = 2, _eta = .025, _seed = 1):
		self.numPrototypes = _numPrototypes - 1
		self.dimensions = _dimensions
		self.eta = _eta
		self.c = int(_numPrototypes*_eta)
		self.seed = _seed
		np.random.seed(_seed)
		self.stateScale = .9 # used to make sure that the prototypes encompass the state space
		self.prototypes = np.random.rand(_numPrototypes, _dimensions)

	def getFeatures(self, _input):
		D = self.prototypes - (np.array(_input)*self.stateScale + np.ones(self.dimensions)*(1-self.stateScale)*.5)
		D = np.sqrt(sum(D.T**2)) # get Euclidian distance
		indexes = np.argpartition(D, self.c, axis=0)[:self.c]
		phi = np.zeros(self.numPrototypes + 1)
		phi[indexes] = 1
		phi[-1] = 1
		return phi

	def removeObs(self, index):
		self.dimensions -= 1
		self.prototypes = np.delete(self.prototypes, index, axis=1)

	def addObs(self):
		raise Exception('To be implemented')
