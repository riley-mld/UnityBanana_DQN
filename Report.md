# Report
## Learning Algorithm
I used a deep Q learning algorithm for this project. Two identical neural network (identical in architecture) were used, one that does the learning and one set as target and soft update happens after a few time-steps of learning to make the target weights closer to the learning neural network. I also used double DQN, a dueling DQN and prioritized experience replay to improve the performance of the Agent. Although i ended up not using prioritized experience replay for the final result, as I wasn't able to get better results using prioritized experience replay.

### Hyper parameters
* Replay Buffer Size = 100000
* Batch Size = 64
* Gamma(Discount factor) = 0.99
* Tau(Soft update parameter) = 0.001
* Learning Rate = 0.0005
* Update Every = 4
* number of episodes = 3000
* Epsilon Start = 1.0
* Epsilon End = 0.01
* Epsilon Decay Rate = 0.995
#### Hyper-parameters for Prioritized Experience Replay (not used for final result)
* A(importance sampling parameter) = 0.6
* B(prioritization parameter) = 0.4
* epsilon(for prioritized replay buffer) = 0.000001

### Model Architecture
`fc1 = [37, 512]`
`fc2 = [512, 128]`
`fc3 = [128, 4]`

## Plot of Rewards
Agent solved the environment in episode 439! And achieved a maximum average score over last 100 episode of 16.21 after training about 1800 episodes.

![](Extras/Graph.png?raw=true)

## Ideas for Future Work
Implementing a Rainbow DQN and attempting to make the agent learn directly from the pixels.
