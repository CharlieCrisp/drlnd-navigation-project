from unityagents import UnityEnvironment
from dqn_agent import Agent
import torch

import matplotlib.pyplot as plt


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


def train(agent, environment, brain_name, fig, ax, line, n_episodes, max_t=1000, eps_start=1.0, eps_decay_rate=0.995, eps_end=0.01):
    eps = eps_start

    scores = []
    indexes = []

    # Define episode timestep
    for i_episode in range(0, n_episodes + 1):
        env_info = environment.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = environment.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
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

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return indexes, scores


def get_action_and_state_size(environment, brain_name):
    brain = environment.brains[brain_name]

    env_info = environment.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size

    state = env_info.vector_observations[0]
    state_size = len(state)
    return action_size, state_size


if __name__ == "__main__":
    env = UnityEnvironment(file_name="Banana.app")

    brain_name = env.brain_names[0]
    action_size, state_size = get_action_and_state_size(env, brain_name)

    training_agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    fig, ax = plt.subplots()
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Agent score')
    line, = ax.plot([], [])
    draw(fig)

    indexes, scores = train(training_agent, env, brain_name, fig, ax, line, n_episodes=1000)

    env.close()

    plt.plot(indexes, scores)
    plt.show()
