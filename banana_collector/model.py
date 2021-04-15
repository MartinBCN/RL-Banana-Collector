import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

torch.manual_seed(42)


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.

    Parameters
    ----------
    state_size: int
        Dimension of each state
    action_size: int
        Dimension of each action
    fc1_units: int
        Number of nodes in first hidden layer
    fc2_units: int
        Number of nodes in second hidden layer
    """

    def __init__(self, state_size: int, action_size: int, fc1_units: int = 64, fc2_units: int = 64) -> None:
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # self.main = nn.Sequential([])

    def forward(self, state: Tensor) -> Tensor:
        """
        Build a network that maps state -> action values.

        Parameters
        ----------
        state: Tensor
            State tensor, shape [batch_size, state_size]
        Returns
        -------
        action: Tensor
            State tensor, shape [batch_size, action_size]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
