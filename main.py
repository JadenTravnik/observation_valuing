import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.monitor.start('q-learning_experiment-2', force=True)

#Q = np.zeros((env.observation_space.n, env.action_space.n)) # initialize Q matrix to zeros

W = np.zeros((5, env.act))

epsilon = 1.0   # probability to take random action
epsilon_decay = 0.98    # probability decays every episode

alpha = 0.1     # learning rate to update Q

num_episodes = 5000

_map = np.array([[0,0,0,0],
                 [0,-1,0,-1],
                 [0,0,0,-1],
                 [-1,0,0,1]])
_map = np.pad(_map, ((1,1),(1,1)), 'constant', constant_values=-1)

# returns the partially observable map: left, up, right, down, center
def cellToArea(cell):
    x = cell / 4 + 1
    y = cell % 4 + 1
    local_map = [_map[x - 1,y],
            _map[x, y - 1],
            _map[x + 1, y],
            _map[x, y + 1],
            _map[x, y]]
    area = []
    for l in local_map:
        if l == -1:
            area.extend([1,0,0])
        elif l == 0:
            area.extend([0,1,0])
        else:
            area.extend([0,0,1])
    return area


from IPython import embed
for i_episode in xrange(num_episodes):
    observation = env.reset()
    done = False

    while not done:
        # current state
        state = observation
        # env.render()

        # choose optimal action
        if np.random.rand() > epsilon:
            #action = np.argmax(Q[state,:])  # choose best action according to current Q matrix


        else:
            action = action = env.action_space.sample()     # random action

        # take action and observe state and reward
        observation, reward, done, info = env.step(action)

    # update Q matrix
    if reward == 0:
        # if we fell in a hole, reward is -100
        R = -100
    else:
        # if we reached goal, reward is 100
        R = 100
    # Q-learning update
    Q[state,action] += alpha * (R + np.max(Q[observation,:]) - Q[state,action])

    # decay epsilon
    epsilon *= epsilon_decay

env.monitor.close()

import matplotlib.pyplot as plt
Q = np.max(Q, axis=1).reshape((4,4))
plt.imshow(Q)
plt.show()
