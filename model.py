import torch
from torch.autograd import Variable
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


class Net1(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):
        super(Net1, self).__init__()

        self.__input_layer = nn.Linear(input_dim, hidden_dim_1)
        self.__hiddem_layer_1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.__hiddem_layer_2 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.__out_layer = nn.Linear(hidden_dim_3, output_dim)

    def forward(self, features):
        device = torch.device("cuda")

        input_to_hidden_1 = F.relu(self.__input_layer(features))
        hidden1_to_hidden2 = F.relu(self.__hiddem_layer_1(input_to_hidden_1))
        hidden2_to_hidden3 = F.relu(self.__hiddem_layer_2(hidden1_to_hidden2))
        hidden3_to_output = self.__out_layer(hidden2_to_hidden3)

        return hidden3_to_output