import torch

from model import DQN_MLP, DDQN_MLP, Net1
from utils import parse_args


def load_mode(path):
    args = parse_args()
    size_board, hidden_dim, output_dim = args.size_board, args.hidden_dim, 4
    device = torch.device("cuda")
    model = DQN_MLP(
        size_board * size_board, hidden_dim, output_dim
    )
    model.load_state_dict(torch.load(path))
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model

    return model


def load_model():
    args = parse_args()
    device = torch.device("cuda")

    size_board, output_dim = args.size_board, 4

    hidden_dim_1 = args.hidden_dim_1
    hidden_dim_2 = args.hidden_dim_2
    hidden_dim_3 = args.hidden_dim_3

    policy_net = Net1(
        size_board * size_board, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim
    )

    policy_net.load_state_dict(torch.load("test2.pth"))
    # policy_net.to(device)
    print(policy_net.state_dict())


# print(load_mode("./best_20000.pth"))
load_model()