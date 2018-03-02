from horde import *
from legion import *
from actorcritic import *
import matplotlib.pyplot as plt
from gridworld import *


def policyTest():

	circle_w = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

	policy = Policy(numFeatures=25, numActions=4, epsilon=0, id=0)
	policy.setWeights(circle_w)

	world = GridWorld(log=True, start=9)
	phi = world.state
	for i in range(0, 30):
		a = policy.getAction(phi)
		phi, r = world.takeAction(a)

def legionTest():
	world = GridWorld(log=True, start=9)

	left_w = np.tile([1, 0, 0, 0], (25, 1))
	up_w = np.tile([0, 1, 0, 0], (25, 1))
	right_w = np.tile([0, 0, 1, 0], (25,1))
	down_w = np.tile([0, 0, 0, 1], (25,1))

	circle_w = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

	legion = PolicyLegion()
	legion.addPolicy(Policy(w=left_w))
	legion.addPolicy(Policy(w=up_w))
	legion.addPolicy(Policy(w=right_w))
	legion.addPolicy(Policy(w=down_w))
	legion.addPolicy(Policy(w=circle_w))

	legion.setBehaviour(4)

	world = GridWorld(log=True, start=9)
	phi = world.state
	for i in range(0, 30):
		a = legion.getAction(phi)
		print(legion.Rho)
		phi, r = world.takeAction(a)

def onPolicyGVFTest(iterations=10000):
	random_w = np.tile([.25], (25, 4))
	policy = Policy(w=random_w)

	world = GridWorld(start=9)
	phi = world.state

	demon = OnPolicyGVF(numFeatures=25,alpha=0.1)

	phi_next =  np.array(np.zeros(25))


	for i in range(0,iterations):
		a = policy.getAction(phi)
		phi_next, r = world.takeAction(a)
		phi_next = np.array(phi_next)

		demon.update(phi, phi_next, r, gamma_t=1, gamma_t_1=1)
		phi = phi_next

	weights = np.flipud(demon.w.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
	    x_val, y_val = int(x_val), int(y_val)
	    ax.text(x_val, y_val, round(weights[y_val][x_val],1), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Value function of random policy')
	plt.show()

def offPolicyGVFTest(iterations=10000):

	circle_w = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
	circle_policy = Policy(w=circle_w)

	random_w = np.tile([.25], (25, 4))
	random_policy = Policy(w=random_w)

	world = GridWorld(start=9)
	phi = world.state

	offDemon = OffPolicyGVF(0, 0, 0, numFeatures=25, lamda=0.9999, alpha=.1/25, beta=.01/25)

	phi_next =  np.array(np.zeros(25))

	for i in range(1,iterations):

		a = random_policy.getAction(phi)
		phi_next, r = world.takeAction(a)
		phi_next = np.array(phi_next)

		rho = circle_policy.getActionProb(a, phi)/random_policy.actionProbs

		if rho > 1:
			rho = 1

		offDemon.update(phi, phi_next, r, rho, 1, 1)
		phi = phi_next

	weights = np.flipud(offDemon.w.reshape((5,5)))
	#visits = np.flipud(world.stateVisits.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
		x_val, y_val = int(x_val), int(y_val)
		ax.text(x_val, y_val, round(weights[y_val][x_val],4), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Value function of Circle policy')
	plt.show()

def onPolicyHordeTest(iterations=10000):

	circle_w = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], \
			[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
			[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
			[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
			[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

	policy = Policy(w=circle_w)

	world = GridWorld(start=9)
	phi = world.state

	horde = Horde()
	horde.addDemon(OnPolicyGVF(numFeatures=25, alpha=0.1, gamma_id=0)) # reward demon
	horde.addDemon(OnPolicyGVF(numFeatures=25, alpha=0.1, gamma_id=1)) # steps till state 24 demon

	c = 0

	for i in range(0,iterations):
		a = policy.getAction(phi)
		phi_next, r = world.takeAction(a)

		Gamma = [.9]

		Z = [r,c]
		if world.stateIndex == 24:# in bottom right
			Gamma.append(1)
			c = 0
		else:
			Gamma.append(0)
			c += 1


		phi_next = np.array(phi_next)
		Phi = [phi]
		Phi_next = [phi_next]


		preds = horde.update(Phi, Phi_next, Z, Gamma, Gamma)
		print(preds)
		phi = phi_next

	weights = np.flipud(horde.demons[0].w.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
	    x_val, y_val = int(x_val), int(y_val)
	    ax.text(x_val, y_val, round(weights[y_val][x_val],1), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Steps until state 4 in Circle policy')
	plt.show()

def offPolicyHordeTest(iterations=10000):
	circle_w = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], \
		[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
	circle_policy = Policy(w=circle_w)

	random_w = np.tile([.25], (25, 4))
	random_policy = Policy(w=random_w)

	world = GridWorld(start=9)
	phi = world.state

	horde = OffPolicyHorde()

	horde.addDemon(OffPolicyGVF(0, 0, 0, numFeatures=25, lamda=0.9999, alpha=.1/25, beta=.01/25))

	phi_next =  np.array(np.zeros(25))

	Gamma = [1]

	for i in range(1,iterations):

		a = random_policy.getAction(phi)
		phi_next, r = world.takeAction(a)
		phi_next = np.array(phi_next)

		Phi = [phi]
		Phi_next = [phi_next]

		Row = [circle_policy.getActionProb(a, phi)/random_policy.actionProbs]

		Z = [r]

		if sum(Row) > 1:
			Row /= sum(Row)

		horde.update(Phi, Phi_next, Z, Row, Gamma, Gamma)
		phi = phi_next

	weights = np.flipud(horde.demons[0].w.reshape((5,5)))
	#visits = np.flipud(world.stateVisits.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
		x_val, y_val = int(x_val), int(y_val)
		ax.text(x_val, y_val, round(weights[y_val][x_val],4), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Off Policy Value function of Circle policy while behaving randomly')
	plt.show()

def DiscreteACTest(iterations=10000):
	dac = DiscreteActorCritic(25, 4, .01, .9, .9, 'discrete_actor')

	world = GridWorld(start=9)
	phi = world.state

	phi_next =  np.array(np.zeros(25))

	totalR = []
	totalPi = []
	totalState = []
	for i in range(0,iterations):
		a, pi = dac.getAction(phi)
		phi_next, r = world.takeAction(a)
		phi_next = np.array(phi_next)

		totalR.append(r)
		totalPi.append(dac.pi.T[0])
		totalState.append(world.stateIndex)

		dac.update(a, phi, phi_next, r)
		phi = phi_next

		if i > 0 and not i % 500:
			print('Iteration ' + str(i) + ': Policy: ' + str(dac.pi.T[0]) + ' avg reward: ' + str(dac.average_r))

	weights = np.flipud(dac.w.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
	    x_val, y_val = int(x_val), int(y_val)
	    ax.text(x_val, y_val, round(weights[y_val][x_val],1), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Value function of random policy')
	plt.show()

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	totalPi = np.array(totalPi)
	for action in range(4):
		ax1.plot(totalPi[:,action], label=world.actionNames[action])

	ax2.plot(totalState, label='State')
	ax3.plot(totalR, label='Reward')

	ax1.legend()
	ax2.legend()
	ax3.legend()
	plt.show()

def ContinuousACTest(iterations=10000):
	dac = ContinuousActorCritic(25, .0001, .9, .9, max_sigma=360, min_mean = 0, max_mean=360)

	world = ContinuousGridWorld(start=9)
	phi = world.state

	phi_next =  np.array(np.zeros(25))

	totalR = []
	totalMean, totalSigma = [], []
	totalState = []
	totalAction = []

	for i in range(0,iterations):
		a = dac.getAction(phi)

		print('Action before ' + str(a))
		angle = a * 360
		angle %= 360

		print('angle afterwards: ' + str(angle))



		phi_next, r = world.takeAction(angle)
		phi_next = np.array(phi_next)

		totalR.append(r)
		totalMean.append(dac.mean)
		totalSigma.append(dac.sigma)
		totalAction.append(round(a))


		totalState.append(world.stateIndex)

		dac.update(a, phi, phi_next, r)
		phi = phi_next

		if i > 0 and not i % 500:
			print('Iteration ' + str(i) + ': mean: ' + str(dac.mean) + ' sigma: ' + str(dac.sigma) + ' avg reward: ' + str(dac.average_r))

	weights = np.flipud(dac.v.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
	    x_val, y_val = int(x_val), int(y_val)
	    ax.text(x_val, y_val, round(weights[y_val][x_val],1), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Value function of random policy')
	plt.show()

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	ax1.plot(totalMean, label='mean')
	ax1.plot(totalSigma, label='sigma')
	ax1.plot(totalAction, label='action')


	ax2.plot(totalState, label='State')
	ax3.plot(totalR, label='Reward')

	ax1.legend()
	ax2.legend()
	ax3.legend()
	plt.show()


def MCACTest():

	dac = MultivariateContinuousActorCritic(25, 2, .0001, .9, .9)

	world = MultiActionContinuousGridWorld(start=9)
	phi = world.state

	phi_next =  np.array(np.zeros(25))

	totalR = []
	totalMean, totalSigma = [], []
	totalState = []
	totalAction = []

	for i in range(0,100000):
		a = dac.getAction(phi)

		print('Action vector ' + str(a))

		phi_next, r = world.takeAction(a)
		phi_next = np.array(phi_next)

		totalR.append(r)
		totalMean.append(dac.mean)
		totalSigma.append(dac.sigma)
		totalAction.append(a)


		totalState.append(world.stateIndex)

		dac.update(a, phi, phi_next, r)
		phi = phi_next

		if i > 0 and not i % 500:
			print('Iteration ' + str(i) + ': mean: ' + str(dac.mean) + ' sigma: ' + str(dac.sigma) + ' avg reward: ' + str(dac.average_r))

	weights = np.flipud(dac.v.reshape((5,5)))

	fig, ax = plt.subplots()
	ax.imshow(weights, interpolation='nearest')

	min_val, max_val, diff = 0., 5., 1.

	ind_array = np.arange(min_val, max_val, diff)
	x, y = np.meshgrid(ind_array, ind_array)

	for x_val, y_val in zip(x.flatten(), y.flatten()):
	    x_val, y_val = int(x_val), int(y_val)
	    ax.text(x_val, y_val, round(weights[y_val][x_val],1), va='center', ha='center')

	#set tick marks for grid
	ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xlim(min_val-diff/2, max_val-diff/2)
	ax.set_ylim(min_val-diff/2, max_val-diff/2)
	ax.grid()

	plt.title('Value function of random policy')
	plt.show()

	f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

	ax1.plot(totalMean, label='mean')
	ax1.plot(totalSigma, label='sigma')
	ax1.plot(totalAction, label='action')


	ax2.plot(totalState, label='State')
	ax3.plot(totalR, label='Reward')

	ax1.legend()
	ax2.legend()
	ax3.legend()
	plt.show()
