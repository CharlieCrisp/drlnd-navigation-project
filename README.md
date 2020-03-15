# Project 1: Navigation
This is a submission for the Udacity Deep Reinforcement Learning Nanodegree navigation project.

## Setup using venv
Create a virtualenv and install the following packages:
 - torch
 - unityagents
 - matplotlib
 
## Training agent
 - `source ./venv/bin/activate`
 - `python train_agent.py --checkpoint-file checkpoint/checkpoint.pth --episodes 500`
 - You will see a graph of the reward that the agent receives over time.
 - Use `python train_agent.py --help` for information about required arguments

## Playing game using trained agent
 - `source ./venv/bin/activate`
 - `python play_game_with_trained_agent.py --checkpoint-file checkpoint/checkpoint.pth`


## Details
This project uses a neural network with 2 hidden layers. 
The two layers have 64 nodes.
The agent uses prioritized experience replay, fixed q targets and an Adam optimiser.

The agent learns to solve the environment with an average reward around 15 in roughly 400 episodes.
Using just uniform random experience replay, the agent achieves the same result in roughly 750 episodes. 

### Prioritized Experience Replay
![Prioritized experience replay](./img/Prioritized%20Experience%20Replay%201000%20eps.png)

### Uniform Random Experience Replay
![Prioritized experience replay](./img/Experience%20Replay%20750%20eps.png)