import argparse

import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent
from train_agent import get_action_and_state_size


def play_game_using_policy(trained_agent, env, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = trained_agent.act(state, 0)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))


def load_agent(action_size, state_size, checkpoint_file):
    print(f"Loading agent weights from file {checkpoint_file}")
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_file))
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a round of banana game using a trained agent.')
    parser.add_argument('--checkpoint-file', type=str, dest='filename', help='The file to load agent weights from', default="checkpoints/checkpoint.pth")

    args = parser.parse_args()
    env = UnityEnvironment(file_name="Banana.app")

    brain_name = env.brain_names[0]
    action_size, state_size = get_action_and_state_size(env, brain_name)

    agent = load_agent(action_size, state_size, args.filename)
    play_game_using_policy(agent, env, brain_name)

    env.close()
