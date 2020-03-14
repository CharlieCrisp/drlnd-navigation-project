# Project 1: Navigation
This is a submission for the udacity Deep Reinforcement Learning Nanodegree navigation project.

## Setup using venv
Create a virtualenv and install the following packages:
 - torch
 - unityagents
 - ipython 
 - jupyter
 
## Running project
Start a jupyter notebook server and navigate to `Navigation.ipynb`.

## Details
This project uses a neural network with 3 hidden layers. 
The first two layers have 64 nodes and the last layer has 32.
The agent learns to solve the environment with an average reward around 15 in roughly 750 episodes.

The agent uses experience replay, fixed q targets and an Adam optimiser.