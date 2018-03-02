import numpy as np

class GridWorld():
	# actions are always left, up, right, down

	def __init__(self, log=False, start = 0):
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[start] = 1
		self.stateIndex = start
		self.log = log
		self.actionNames = ['left', 'up', 'right', 'down']

	def reset(self, _state = 0):
		self.stateIndex = _state
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[_state] = 1

	def takeAction(self, a):
		oldStateIndex = self.stateIndex
		self.state[self.stateIndex] = 0
		r = 0
		if self.stateIndex == 1: # A
			self.stateIndex = 16
			r = 10
		elif self.stateIndex == 3: # B
			self.stateIndex = 13
			r = 5
		else:
			if a == 0: # left
				if self.stateIndex % 5 == 0:
					r = -1
				else:
					self.stateIndex -= 1
			elif a == 1: # up
				if self.stateIndex < 5:
					r = -1
				else:
					self.stateIndex -= 5
			elif a == 2: # right
				if self.stateIndex % 5 == 4:
					r = -1
				else:
					self.stateIndex += 1
			elif a == 3: # down
				if self.stateIndex > 19:
					r = -1
				else:
					self.stateIndex += 5

		self.stateVisits[self.stateIndex] += 1
		self.state[self.stateIndex] = 1
		assert(sum(self.state) == 1)

		if self.log:
			print('State ' + str(oldStateIndex) + ' -> ' + self.actionNames[a] + ' -> ' + str(self.stateIndex) + ', ' + str(r))


		return (self.state, r)

class ObservationValuingHallwayWorld():
	def __init__(self, log=False, num_states=7, flip=False):
		self.num_states = num_states
		self.flip = flip
		self.log = log
		self.reset()
	def reset(self, _state=-1):
		if _state < 0:
			_state = np.random.randint(0, self.num_states)
		self.state = _state
		self.goal_end = 0 if self.flip and np.random.random() > .5 else (self.num_states - 1)
		if self.log:
			print('state', self.state, 'goal_end', self.goal_end, 'num_states', self.num_states)

	def takeAction(self, a):
		self.state += -1 + a
		self.state = max(self.state, 0)
		self.state = min(self.state, self.num_states - 1)
		# 0 is left, 1 is stay, 2 is right
		return (self.state, -1, self.state == self.goal_end)

	def getObservation(self, index):
		if type(index) == type(0):
			if index == 0:
				return abs(self.state - self.goal_end)/float(self.num_states - 1)
			else:
				return np.random.random()
		else:
			length = len(index)
			obs = []
			if 0 in index:
				obs = [abs(self.state - self.goal_end)/float(self.num_states - 1)]
				length -= 1
			for i in range(length):
				obs.append(np.random.random())
			return obs


class ContinuousGridWorld():
	# actions are always left, up, right, down

	def __init__(self, log=False, start = 0):
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[start] = 1
		self.stateIndex = start
		self.log = log

	def reset(self, _state = 0):
		self.stateIndex = _state
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[_state] = 1

	def takeAction(self, a):
		oldStateIndex = self.stateIndex
		self.state[self.stateIndex] = 0
		r = 0
		if self.stateIndex == 1: # A
			self.stateIndex = 16
			r = 10
		elif self.stateIndex == 3: # B
			self.stateIndex = 13
			r = 5
		else:

			if 135. < a < 225.: # right
				if self.stateIndex % 5 == 4:
					r = -1
				else:
					self.stateIndex += 1

			elif 45. < a < 135.: # up
				if self.stateIndex < 5:
					r = -1
				else:
					self.stateIndex -= 5
			elif 225. < a < 315.: # down
				if self.stateIndex > 19:
					r = -1
				else:
					self.stateIndex += 5
			else: # left
				if self.stateIndex % 5 == 0:
					r = -1
				else:
					self.stateIndex -= 1

			print('Action angle: ' + str(a))


		self.stateVisits[self.stateIndex] += 1
		self.state[self.stateIndex] = 1
		assert(sum(self.state) == 1)

		if self.log:
			print('State ' + str(oldStateIndex) + ' -> ' + self.actionNames[a] + ' -> ' + str(self.stateIndex) + ', ' + str(r))


		return (self.state, r)



class MultiActionContinuousGridWorld():
	# actions are always left, up, right, down

	def __init__(self, log=False, start = 0):
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[start] = 1
		self.stateIndex = start
		self.log = log

	def reset(self, _state = 0):
		self.stateIndex = _state
		self.state = np.zeros(25)
		self.stateVisits = np.zeros(25)
		self.state[_state] = 1

	def takeAction(self, a):
		oldStateIndex = self.stateIndex
		self.state[self.stateIndex] = 0
		r = 0
		if self.stateIndex == 1: # A
			self.stateIndex = 16
			r = 10
		elif self.stateIndex == 3: # B
			self.stateIndex = 13
			r = 5
		else:

			# a[0] = x, a[1] = y

			if -.1 < a[0] < .1:  # ~0
				if self.stateIndex % 5 == 4:
					r = -1
				else:
					self.stateIndex += 1

			if .9 < a[1] < 1.1: # up ~1
				if self.stateIndex < 5:
					r = -1
				else:
					self.stateIndex -= 5

			if -.1 < a[1] < .1: # down
				if self.stateIndex > 19:
					r = -1
				else:
					self.stateIndex += 5
			if .9 < a[0] < 1.1: # left
				if self.stateIndex % 5 == 0:
					r = -1
				else:
					self.stateIndex -= 1

			print('Action angle: ' + str(a))


		self.stateVisits[self.stateIndex] += 1
		self.state[self.stateIndex] = 1
		assert(sum(self.state) == 1)

		if self.log:
			print('State ' + str(oldStateIndex) + ' -> ' + self.actionNames[a] + ' -> ' + str(self.stateIndex) + ', ' + str(r))


		return (self.state, r)
