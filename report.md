# Report
## Learning Algorithm
	
The underlying Deep-Q Network consists of two hidden layers of size 64 with ReLU activation.
Input and output size are defined by the environment as 37 and 4, respectively.
As an alternative a Dueling Deep-Q Network was tested where the outputs of the second hidden
layer are routed through a value- and an advantage-layer before being aggregated again. 

For the training algorithm a variety of approaches was tested:
* simple DQN learning where the 

## Plot of Rewards
	
![alt text](figures/scores.png "Scores")

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

## Ideas for Future Work
The following tasks are partially or fully open:
* Fix the Double-Q implementation: the current one does not seem to learn fast enough;
  either the implementation is wrong, or it needs improvement
* Include prioritized experience replay
* Use (probably two) convolution layers to learn from the graphics output rather than the 
    environment state
* Proper hyper-parameter training: since even the most basic DQN implementation comes close 
    to solving this problem a hyper-parameter tuning was skipped all-together in favor 
    of an exploration of the different DQN variations. This would include exploring different
  layer sizes, learning rates, discount factors and others