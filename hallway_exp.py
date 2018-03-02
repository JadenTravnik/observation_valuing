import numpy as np
import matplotlib.pyplot as plt
from gridworld import ObservationValuingHallwayWorld
from actorcritic import DiscreteActorCritic
from representation import SelectiveKanervaCoder


world = ObservationValuingHallwayWorld(num_states=30)

num_episodes = 500
num_runs = 20
TotalReward = []
noise = [0, 1, 2, 4, 8, 16, 32, 64, 128]
features = 1000
max_steps = 150

for _noise in noise:

    TotalReward.append([])
    skc = SelectiveKanervaCoder(features, _dimensions = _noise + 1)

    for run_i in range(num_runs):
        TotalReward[-1].append([])
        agent = DiscreteActorCritic(features, 2, .01, .3, 1, 'dac', w_lambda=None)
        if not run_i % int(num_runs*.1):
            print('Noise ' + str(_noise) + ' Run ' + str(run_i))
        for episode_i in range(num_episodes):

            TotalReward[-1][-1].append(0)
            steps = 0
            world.reset()
            phi_t = skc.getFeatures(world.getObservation(range(_noise + 1)))
            action, _ = agent.getAction(phi_t)

            while True:
                steps += 1
                state, reward, terminal = world.takeAction(action*2)
                TotalReward[-1][-1][-1] += 1
                phi_t_1 = skc.getFeatures(world.getObservation(range(_noise + 1)))
                agent.update(action, phi_t, phi_t_1, reward)
                if terminal or steps == max_steps:
                    break

                action, pi = agent.getAction(phi_t_1)
                phi_t = phi_t_1

def running_mean(arr, window):
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    if window == 0:
        return cumsum / float(len(cumsum))
    elif window > 0:
        return (cumsum[window:] - cumsum[:-window]) / window
    else:
        return arr / float(len(arr))
TotalReward = np.array(TotalReward)

np.save('oneway_hallway', TotalReward)

TotalReward = np.mean(TotalReward, axis=1)
for i in range(len(noise)):
    plt.plot(TotalReward[i,:], label=str(noise[i]) + ' signals')

plt.ylabel('Steps to Complete Episode')
plt.xlabel('Episodes')
plt.legend()
plt.show()
