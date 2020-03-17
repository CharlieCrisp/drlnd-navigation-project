# Project report
This project uses a neural network with 3 hidden layers. 
The first two layers have 64 nodes and the last has 32.
The agent uses prioritized experience replay, fixed q targets and an Adam optimiser.

For the pixel challenge where the agent is trained from a 84x84 RGB input image, the project uses two convolutional 
layers with maxpooling, two fully connected layers and a dropout layer.

The agent learns to solve the environment in 455 episodes. 
This means it gets an average score of over 13 (averaged over 100 consecutive episodes).
Using just uniform random experience replay, the agent solves the environment in 463 episodes.
The difference here is minimal but with more tuning of hyperparameters, the agent may even solve the environment more quickly.
For more information about the uniform experience replay, please see the branch `release/uniform-experience-replay`.

## Hyperparameters
The project uses the following hyperparameters:
```python
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

A = 0.6                 # The value of alpha for prioritized experience replay
B_START = 0.4           # The initial value of beta for prioritized experience replay
BETA_BATCH_NOS = 50_000 # The number of samples over which to anneal beta to 1.0

eps_start = 1.0         # The initial value of epsilon
eps_decay_rate = 0.992  # The rate with which epsilon decays every timestep
eps_end = 0.01          # The minimal value of epsilon

```


## Learning Algorithm
The project uses the [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) learning algorithm which is a 
gradient based optimisation algorithm for stochastic objective functions. 
 
## Results
### Prioritized Experience Replay
![Prioritized experience replay](./img/Prioritized%20Experience%20Replay%20750%20eps.png)

### Uniform Random Experience Replay
This is the results of the project before Prioritized Experience Replay was added.
![Prioritized experience replay](./img/Experience%20Replay%20750%20eps.png)

## Future improvements
There are a few ways in which I would like to improve this solution to the project. 

### More hyperparameter training for visual input
I have not been able to train the visual agent on a GPU and training on a CPU is too slow.
I would like to be able to experiment with some of the hyperparameters of the network and try a few different CNN 
architectures, whilst training on a GPU so I could get a viable solution.

### Double DQN
Because of the step in learning where we calculate the TD error using the max over the noisy target action value estimations, 
we mostly end up overestimating the Q-value. 
Double DQN is a method that calculates the optimal action using one set of weights (e.g. local weights) and then evaluates
the Q-value of this action using separate weights (e.g. target weights).
This makes it more likely that this noise will cancel rather than add and can help agents learn quicker.

### Dueling DQN
Dueling DQN refers to an algorithm where the action value is split into the sum of the state value (V) and the 
'advantage' (A) of a state action pair. 
The advantage is how valuable an action in a state is compared to the other actions.
The algorithm is trained to learn V and A separately and this allows it to learn which states are high value, regardless
of the subsequent actions.
This is beneficial for two reasons:
 - If V is high, we don't need to calculate Q for each action because our action will not matter much.
 This can help accelerate learning.
 - We can get more reliable estimates for Q because we're decoupling estimates for the two values.