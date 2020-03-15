import argparse
import os
import torch

import matplotlib.pyplot as plt

from os import path
from unityagents import UnityEnvironment
from dqn_agent import Agent


plt.ion()


def draw(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_scores(times, scores, fig, ax, line):
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(min(scores), max(scores))

    line.set_xdata(times)
    line.set_ydata(scores)
    draw(fig)


def train(agent, environment, brain_name, fig, ax, line, checkpoint_filename, n_episodes, get_state, max_t=1000, eps_start=1.0, eps_decay_rate=0.995, eps_end=0.01):
    print(f"Trained agent weights will be saved to {checkpoint_filename}")
    print(f"Agent will be trained for {n_episodes} episodes")
    eps = eps_start

    scores = []
    indexes = []

    # Define episode timestep
    for i_episode in range(0, n_episodes + 1):
        env_info = environment.reset(train_mode=True)[brain_name]
        state = get_state(env_info)

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = environment.step(action)[brain_name]
            next_state = get_state(env_info)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            agent.step(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        scores.append(score)
        indexes.append(i_episode)
        plot_scores(indexes, scores, fig, ax, line)

        eps = max(eps_end, eps * eps_decay_rate)

    checkpoint_dir = path.dirname(checkpoint_filename)
    if checkpoint_dir != '' and not path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.save(agent.qnetwork_local.state_dict(), checkpoint_filename)
    return indexes, scores


def get_action_and_state_size(environment, brain_name, is_visual):
    brain = environment.brains[brain_name]

    env_info = environment.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size

    if is_visual:
        return action_size, None
    else:
        state = env_info.vector_observations[0]
        state_size = len(state)
        return action_size, state_size


def get_vector_state(env_info):
    return env_info.vector_observations[0]


def get_visual_state(env_info):
    return env_info.visual_observations[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent to play the banana game.')
    parser.add_argument('--checkpoint-file', type=str, dest='filename', help='The file in which to save agent weights', default="checkpoints/checkpoint.pth")
    parser.add_argument('--episodes', type=int, dest='episodes', help='Number of episodes to train agent for', default=1000)
    parser.add_argument('--visual', dest='visual', help='Train agent on only input pixels rather than preprocessed state vector', action='store_true')

    args = parser.parse_args()

    env = UnityEnvironment(file_name="Banana.app") if not args.visual else UnityEnvironment(file_name="VisualBanana.app")

    brain_name = env.brain_names[0]
    action_size, state_size = get_action_and_state_size(env, brain_name, args.visual)

    training_agent = Agent(state_size=state_size, action_size=action_size, seed=0, visual=args.visual)

    get_state = get_vector_state if not args.visual else get_visual_state

    fig, ax = plt.subplots()
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Agent score')
    line, = ax.plot([], [])
    draw(fig)

    indexes, scores = train(training_agent, env, brain_name, fig, ax, line, args.filename, n_episodes=args.episodes, get_state=get_state)

    plt.savefig("trained_agent.png")
    env.close()
