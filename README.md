# Deep Reinforcement Learning Agent for Collecing Bananas
This project was done as part of the _Udacity Deep Reinforcement Learning Nanodegree_.
Deep Q Learning was used in this project to teach an agent to collect yellow bananas and avoid blue bananas in a Unity environment.


## Get Started
For Installing this code you will need to install the dependencies first or You can also install the folder python in the repo using git for easy installation.

### Dependencies
`Pytorch, Numpy, Matplotlib, unityagents, Jupyter Notebook`


## Instructions
For Training the agents after going through the installation steps, first open the Navigation file with Jupyter Notebook. Then change the line with the path to the enviroment, to where it's located in your computer. You can download the files from here, and unzip them at the repositories location.
* Linux: [click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

After this run the cells in order.

## Project Details
In this enviroment the action space was 4, and the state space was 37. I intend to add a version of the agent that would use the raw image output as state space instead. 


## Deep Q Learning
For this project I used and experimented with two identical network one as a target one as the network that trains.
I Also used techniques such as Double DQN, Dueling DQN and Priorotised Experience Replay.
For more information read the report file in the repo


## License
This project is license under the MIT License. For more info check the license file in the repo. 
