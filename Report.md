# Report

## Learning Algorithm
I used a deep Q learning algorithm for this project. Two identical neural network (identical in archeticture) were used, one that does the learning and one set as target and soft update happenes after a few timesteps of learning to make the target weights closer to the learning neural network. 

### Hyper parameters
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR = 5e-4               # learning rate 
* UPDATE_EVERY = 4        # how often to update the network
* PRB_A = 0.6             # importance sampling parameter
* PRB_B = 0.4             # prioritization parameter
* max_t = 1000
* eps = 1e-6 for priotoised replay buffer
* n_episodes = 3000
* eps_start = 1.0
* eps_end = 0.01
* eps_decay = 0.995

## Plot of Rewards

## Ideas for Future Work
