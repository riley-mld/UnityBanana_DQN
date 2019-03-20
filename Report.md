# Report

## Learning Algorithm
I used a deep Q learning algorithm for this project. Two identical neural network (identical in archeticture) were used, one that does the learning and one set as target and soft update happenes after a few timesteps of learning to make the target weights closer to the learning neural network. I also used double DQN, a dueling DQN and prioritized experience replay to improve the performence of the Agent. Although i ended up not using prioritized experience replay for the final result, as I wasn't able to get better results using prioritized experience replay.

### Hyper parameters
* Replay Buffer Size = 100000
* Batch Size = 64
* Gamma(Discount factor) = 0.99
* Tau(Soft update parameter) = 1e-3
* Learning Rate = 5e-4
* UPDATE_EVERY = 4        # how often to update the network
* PRB_A = 0.6             # importance sampling parameter
* PRB_B = 0.4             # prioritization parameter
* max_t = 1000
* eps = 1e-6 for priotoised replay buffer
* n_episodes = 3000
* eps_start = 1.0
* eps_end = 0.01
* eps_decay = 0.995

### Model Architecture

## Plot of Rewards

## Ideas for Future Work
