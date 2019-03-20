# Deep Reinforcement Learning Agent for Collecting Bananas
This project was done as part of the [_Udacity Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
Deep Q Learning was used in this project to teach an agent to collect yellow bananas and avoid blue bananas in a Unity environment.

![](Extras/TrainedAgent.gif?raw=true)

## Getting Started
For Installing this code, you will need to install the dependencies first or You can also install the folder python in the repo using git for easy installation.

### Dependencies
`Pytorch, Numpy, Matplotlib, unityagents, Jupyter Notebook`


## Instructions
For Training the agents after going through the installation steps, first open the Navigation file with Jupyter Notebook. Then change the line with the path to the environment, to where it's located in your computer. You can download the files from here and unzip them at the repositories location.
* Linux: [click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Place the file in the DRLND GitHub repository, in the p1_navigation/ folder, and unzip (or decompress) the file.

**(For Windows users)** Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

**(For AWS)** If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

After this Just run the cells in order in `Navigation_NB.ipynb` to train your agent.


## Project Details
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* `0` - move forward.
* `1` - move backward.
* `2` - turn left.
* `3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Deep Q Learning
For this project I used and experimented with two identical network one as a target one as the network that trains.
I Also used techniques such as Double DQN, Dueling DQN and Prioritized Experience Replay.
For more information read the report file in the repo
This papers were used in creating this project:
* [Human-level control through deep reinforcementlearning](http://files.davidqiu.com//research/nature14236.pdf)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)


## License
This project is license under the MIT License. For more info check the license file in the repo.
