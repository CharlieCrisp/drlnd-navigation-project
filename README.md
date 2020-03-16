# Project 1: Navigation
This is a submission for the Udacity Deep Reinforcement Learning Nanodegree navigation project.

## Environment
In this environment, an agent navigates the world trying to collect yellow bananas and avoid blue bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Setup using venv
Create a virtualenv and install the following packages:
 - torch
 - unityagents
 - matplotlib
 
## Training agent
 - `source ./venv/bin/activate`
 - `python train_agent.py --checkpoint-file checkpoint.pth --episodes 500`
 - You will see a graph of the reward that the agent receives over time.
 - Use `python train_agent.py --help` for information about required arguments

## Playing game using trained agent
 - `source ./venv/bin/activate`
 - `python play_game_with_trained_agent.py --checkpoint-file checkpoint.pth`

## Using visual pixel input
Both training and playing scripts take an optional flag `--visual` which causes them to use the visual pixel input version of the problem.
In this scenario, the agent learns to play the game based on a 84x84 RGB pixel input stream rather than a preprocessed vector of details about the game.
The agent must first learn important features and then use them to guide optimal actions.
Note: due to not training with a GPU, there are no saved weights for this version of the network.

## Details
For a write up of the details and results, see the [Report.md](./Report.md) document.