from pathlib import Path

from unityagents import UnityEnvironment
import os
from agent import Agent
from training import Trainer


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

    trainer = Trainer(env,
                      brain=0,
                      max_t=max_t)

    state_size, action_size = trainer.get_sizes()

    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  eps_start=eps_start,
                  eps_end=eps_end,
                  eps_decay=eps_decay,
                  gamma=gamma,
                  tau=tau,
                  lr=lr,
                  update_every=update_every
    )

    trainer.train(agent, n_episodes=n_episodes)
    fn = Path(os.environ.get('FIG_DIR', 'figures')) / 'training_results.png'
    trainer.plot(fn)

    if trainer.solved:
        agent.save('test.pth')


if __name__ == '__main__':

    dqn(n_episodes=2000)
