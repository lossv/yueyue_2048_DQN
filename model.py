import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN_MLP, self).__init__()
        self.__input_layer = nn.Linear(input_dim, hidden_dim)
        self.__hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.__output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        input_to_hidden = F.relu(self.__input_layer(features))
        hidden_to_output = F.relu(self.__hidden_layer(input_to_hidden))
        output = self.__output_layer(hidden_to_output)

        return output


# Dueling DQN
class DDQN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DDQN_MLP, self).__init__()
        self.__input_layer = nn.Linear(input_dim, hidden_dim)
        self.__hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.__advantage_layer = nn.Linear(hidden_dim, output_dim)
        self.__value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        input_to_hidden = F.relu(self.__input_layer(features))
        hidden_to_output = F.relu(self.__hidden_layer(input_to_hidden))
        adv = self.__advantage_layer(hidden_to_output)
        val = self.__value_layer(hidden_to_output)

        output = val + adv - adv.unsqueeze(0).mean(1, keepdim=True)

        return output.squeeze(0)
