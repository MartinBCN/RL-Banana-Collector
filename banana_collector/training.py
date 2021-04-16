from collections import deque
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from unityagents import UnityEnvironment
from agent import Agent
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Trainer:
    """
    Training of DQN

    Parameters
    ----------
    env: UnityEnvironment
        Environment
    brain: int
        Number of brain to be used
    max_t: int, default = 1000
        maximum number of time steps per episode
    """
    def __init__(self, env: UnityEnvironment, brain: int = 0, max_t: int = 1000):

        self.max_t = max_t

        # Environment
        self.env = env
        self.brain_name = env.brain_names[brain]
        _ = env.reset(train_mode=True)[self.brain_name]

        # Book-keeping
        self.solved = False
        self.logger = {"score": [], 'scores_window': deque(maxlen=100), 'loss': []}

    def get_sizes(self) -> Tuple[int, int]:
        """
        Get state size and action size

        Returns
        -------
        state_size: int
        action_size: int
        """
        brain = self.env.brains[self.brain_name]
        state_size = brain.vector_observation_space_size
        action_size = brain.vector_action_space_size
        return state_size, action_size

    def log_scalar(self, name: str, value: Union[int, float]):
        if name in self.logger.keys():
            self.logger[name].append(value)
        else:
            self.logger[name] = [value]
        # ToDo: Tensorboard

    def train(self, agent: Agent, n_episodes: int) -> None:
        """
        Main training loop

        Parameters
        ----------
        agent: Agent
        n_episodes: int

        Returns
        -------
        None
        """

        for i_episode in range(1, n_episodes + 1):

            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state

            score = 0
            episode_loss = 0

            for t in range(self.max_t):
                action = agent.act(state)
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                loss = agent.step(state, action, reward, next_state, done)
                if loss is not None:
                    self.log_scalar('loss_batch', loss)
                    episode_loss += loss
                state = next_state
                score += reward
                if done:
                    break

            # Decrease epsilon
            agent.step_epsilon()

            # Book-keeping
            self.log_scalar('score', score)
            self.log_scalar('score_window', score)
            self.log_scalar('loss', episode_loss)

            mean_score = np.mean(self.logger['score_window'])
            log_str = f'\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}'
            print(log_str, end="")
            if i_episode % 100 == 0:
                print(log_str)

            # Solving Criterion
            if mean_score >= 13.0:  # Criterion in Rubric
                print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {mean_score:.2f}')
                self.solved = True
                break

    def plot(self, file_name: Union[str, Path] = None) -> None:
        """
        Create analysis plot

        Parameters
        ----------
        file_name: Union[str, Path], default = None

        Returns
        -------
        None
        """

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- Scores ---
        scores = self.logger['score']
        x = np.arange(len(scores))
        axes[0].plot(x, scores)
        y = np.convolve(scores, np.ones(100) / 100, mode='same')
        axes[0].plot(x, y)

        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Episode #')

        # --- Losses ---
        losses = self.logger['loss']
        x = np.arange(len(losses))
        axes[1].plot(x, losses)
        y = np.convolve(losses, np.ones(100) / 100, mode='same')
        axes[1].plot(x, y)

        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Episode #')

        if file_name is None:
            fig.show()
        else:
            if type(file_name) is str:
                file_name = Path(file_name)
            file_name.parents[0].mkdir(parents=True, exist_ok=True)
            fig.savefig(file_name)
