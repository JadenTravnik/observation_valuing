import numpy as np
import matplotlib.pyplot as plt
from gridworld import ObservationValuingHallwayWorld
from actorcritic import DiscreteActorCritic
from representation import SelectiveKanervaCoder


world = ObservationValuingHallwayWorld(num_states=30)

num_episodes = 2000
num_runs = 5
TotalReward = []
noise = [9]
features = 1000
max_steps = 150

TotalPi = []
for _noise in noise:

    TotalReward.append([])
    TotalPi.append([])
    skc = SelectiveKanervaCoder(features, _dimensions = _noise + 1)

    for run_i in range(num_runs):
        TotalReward[-1].append([])
        TotalPi[-1].append([])
        agent = DiscreteActorCritic(features, 2, .01, .3, 1, 'dac', w_lambda=None)
        obs_agent = DiscreteActorCritic(features, _noise + 1, .01, .3, 1, 'dac', w_lambda=None)
        if not run_i % 2:
            print('Noise ' + str(_noise) + ' Run ' + str(run_i))
        for episode_i in range(num_episodes):

            TotalReward[-1][-1].append(0)
            steps = 0
            world.reset()


            obs_phi_t = skc.getFeatures(world.getObservation(range(_noise + 1)))
            obs_action, obs_pi = obs_agent.getAction(obs_phi_t)
            phi_t = skc.getFeatures(world.getObservation(obs_action))
            action, _ = agent.getAction(phi_t)

            while True:
                steps += 1
                state, reward, terminal = world.takeAction(action*2)
                TotalReward[-1][-1][-1] += 1
                obs_phi_t_1 = skc.getFeatures(world.getObservation(range(_noise + 1)))
                obs_agent.update(obs_action, obs_phi_t, obs_phi_t_1, reward)

                phi_t_1 = skc.getFeatures(world.getObservation(obs_action))
                agent.update(action, phi_t, phi_t_1, reward)
                if terminal or steps == max_steps:
                    break

                action, pi = agent.getAction(phi_t_1)

                phi_t = phi_t_1
                obs_phi_t = obs_phi_t_1

            TotalPi[-1][-1].append(obs_pi.flatten())

def running_mean(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    if window == 0:
        return cumsum / float(len(cumsum))
    elif window > 0:
        return (cumsum[window:] - cumsum[:-window]) / window
    else:
        return arr / float(len(arr))
TotalReward = np.array(TotalReward)
TotalReward = np.mean(TotalReward, axis=1)
TotalPi = np.array(TotalPi)
TotalPi = TotalPi[0]
t = np.mean(TotalPi, axis=0)

for i in range(t.shape[1]):
    plt.plot(t[:,i], label='signal ' + str(i))

plt.legend()
plt.show()

from IPython import embed
embed()

for i in range(len(noise)):
    plt.plot(TotalReward[i,:], label=str(noise[i]) + ' signals')

plt.ylabel('Steps to Complete Episode')
plt.xlabel('Episodes')
plt.legend()
plt.show()
