import numpy as np
import matplotlib.pyplot as plt
from gridworld import ObservationValuingHallwayWorld
from actorcritic import DiscreteActorCritic
from representation import SelectiveKanervaCoder


world = ObservationValuingHallwayWorld(num_states=30)

num_episodes = 3000
num_runs = 1
TotalReward = []
noise = [20]
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
        obs_agent = DiscreteActorCritic(1, _noise + 1, .1, 0, 1, 'dac', w_lambda=None)

        if num_runs > 1 and not run_i % int(num_runs*.1):
            print('Noise ' + str(_noise) + ' Run ' + str(run_i))
        for episode_i in range(num_episodes):

            steps = 0
            world.reset()

            obs = world.getObservation(range(_noise + 1))
            _, obs_weights = obs_agent.getAction(np.ones(1))
            if not episode_i % 20 and episode_i > 0:
                print('Episode ' + str(episode_i) + ' ' + str(TotalReward[-1][-1][-1]))
            phi_t = skc.getFeatures(obs)
            action, _ = agent.getAction(phi_t)
            obs_action, obs_weights = obs_agent.getAction(np.ones(1))

            TotalReward[-1][-1].append(0)
            while True:
                steps += 1
                state, reward, terminal = world.takeAction(action*2)
                TotalReward[-1][-1][-1] += 1

                obs = world.getObservation(range(_noise + 1))

                obs *= obs_weights.flatten()

                phi_t_1 = skc.getFeatures(obs)
                agent.update(action, phi_t, phi_t_1, world.getObservation(0))

                TotalPi[-1][-1].append(obs_weights)

                # if max(obs_weights) > .99:
                #     if np.argmax(obs_weights) == 0:
                #         print('Elimintated signal')
                #         exit()
                #     worst = np.argmax(obs_weights)
                #     print('noise elimited ' + str(worst) + '. ' + str(_noise) + ' signals left')
                #     _noise -= 1
                #     obs_agent = DiscreteActorCritic(1, _noise + 1, .01, 1, 1, 'dac', w_lambda=None)
                #     skc.removeObs(worst)

                if terminal or steps == max_steps:
                    obs_agent.update(obs_action, np.ones(1), np.ones(1), -TotalReward[-1][-1][-1])
                    break

                action, pi = agent.getAction(phi_t_1)
                phi_t = phi_t_1
        print(obs_weights)

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

for i in range(len(noise)):
    plt.plot(TotalReward[i,:], label=str(noise[i]) + ' signals')

plt.ylabel('Steps to Complete Episode')
plt.xlabel('Episodes')
plt.legend()
plt.show()
