# Report
## Learning Algorithm
	

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

## Plot of Rewards
	

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

## Ideas for Future Work
The following tasks are partially or fully open:
* Fix the Double-Q implementation: the current one does not seem to learn
* Include prioritized experience replay
* Use (probably two) convolution layers to learn from the graphics output rather than the 
    environment state
* Proper hyper-parameter training: since even the most basic DQN implementation comes close 
    to solving this problem a hyper-parameter tuning was skipped all-together in favor 
    of an exploration of the different DQN variations. This would include exploring different
  layer sizes, learning rates, discount factors and others