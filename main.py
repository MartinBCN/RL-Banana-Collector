from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import Agent


def dqn(n_episodes: int = 2000, max_t: int = 1000, eps_start: float = 1.0,
        eps_end: float = 0.01, eps_decay: float = 0.995, buffer_size: int = int(1e5), batch_size: int = 64,
        gamma: float = 0.99, tau: float = 1e-3, lr: float = 5e-4, update_every: int = 4) -> None:
    """
    Training of DQN

    Parameters
    ----------
    n_episodes: int, default = 2000
        maximum number of training episodes
    max_t: int, default = 1000
        maximum number of time steps per episode
    eps_start: float, default = 1.0
        starting value of epsilon, for epsilon-greedy action selection
    eps_end: float = 0.01
        minimum value of epsilon
    eps_decay: float = 0.995
        multiplicative factor (per episode) for decreasing epsilon
    buffer_size: int = int(1e5)
        replay buffer size
    batch_size: int = 64
        mini-batch size
    gamma: float = 0.99
        discount factor
    tau: float = 1e-3
        for soft update of target parameters
    lr: float = 5e-4
        Learning rate
    update_every: int = 4
        how often to update the network

    Returns
    -------
    None
    """
    env = UnityEnvironment('data/Banana_Linux/Banana.x86_64', no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    agent = Agent(state_size=brain.vector_observation_space_size,
                  action_size=brain.vector_action_space_size,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  gamma=gamma,
                  tau=tau,
                  lr=lr,
                  update_every=update_every
    )

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            loss = agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        mean_score = np.mean(scores_window)
        log_str = f'\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}'
        print(log_str, end="")
        if i_episode % 100 == 0:
            print(log_str)
        if mean_score >= 200.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {mean_score:.2f}')
            torch.save(agent.q_network_local.state_dict(), 'checkpoint.pth')
            break

    # --- plot the scores ---
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # --- save the model ---


if __name__ == '__main__':
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # mini-batch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network

    dqn(n_episodes=200)
