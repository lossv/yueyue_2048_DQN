import torch

from model import DQN_MLP, DDQN_MLP
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


print(load_mode("./best_20000.pth"))