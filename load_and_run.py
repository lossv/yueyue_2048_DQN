from copy import deepcopy

import pygame
import torch
import numpy as np
from torch import autograd

from env import Game2048Env
from frame_2048 import draw
from utils import parse_args, plot_info

from model import DQN_MLP, DDQN_MLP


class Agent_Not_Train:
    def __init__(
            self,
            size_board,
            hidden_dim,
            output_dim,
            path
    ):
        self.__use_cuda = torch.cuda.is_available()
        self.__path = path

        self.hidden_dim = hidden_dim
        self.size_board = size_board
        self.output_dim = output_dim

        print("Creating a no-dueling DQN")
        self.__target_net = DQN_MLP(size_board * size_board, hidden_dim, output_dim)
        if self.__use_cuda:
            self.__target_net = self.__target_net.cuda()

        self.__epsilon_threshold = 0.1

        self.__variable = (
            lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            if self.__use_cuda
            else autograd.Variable(*args, **kwargs)
        )

    def __load_(self):
        device = torch.device("cuda")

        self.__target_net.load_state_dict(torch.load(self.__path))
        self.__target_net.to(device)
        # Make sure to call input = input.to(device) on any input tensors that you feed to the model

    def selection_action(self, valid_movements, state):
        sample = np.random.rand(1)

        if sample > self.__epsilon_threshold:
            with torch.no_grad():
                output = self.__target_net(
                    self.__variable(torch.from_numpy(state).float())
                )
                output_np = output.cpu().detach().numpy()
                ordered = np.flip(
                    np.argsort(np.expand_dims(output_np, axis=0), axis=1)
                )[0]
                for action in ordered:
                    if valid_movements[action] != 0:
                        return action

        else:
            return np.random.choice(np.nonzero(valid_movements)[0])


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


def play(env, agent, episodes, interval_mean, screen):

    rewards_per_episode = []
    loss_per_episode = []
    steps_per_episode = []
    scores_per_episode = []
    threshold = []
    decay_step = 0
    best_score = 0
    best_ep = 0
    best_board = 0
    best_steps = 0
    best_reward = 0
    ep = 0
    for ep in range(episodes):
    # while True:
        print(ep)

        done = 0
        state, valid_movements = env.reset()
        loss_ep = []
        episode_rewards = []
        steps = 0

        while True:

            done = 0
            draw(state.flatten(), screen)
            steps += 1

            action = agent.selection_action(valid_movements, state.flatten())
            next_state, reward, done, info = env.step(action)

            if done == 1:

                steps_per_episode.append(steps)

                rewards_per_episode.append(np.sum(episode_rewards))
                loss_per_episode.append(np.sum(loss_ep) / steps)
                scores_per_episode.append(info["total_score"])

                if info["total_score"] > best_score:
                    best_score = info["total_score"]
                    best_reward = np.sum(episode_rewards)
                    best_ep = ep
                    best_board = deepcopy(next_state)
                    best_steps = steps
                    print(best_board)

            else:

                state = deepcopy(next_state)

                valid_movements = info["valid_movements"]

            if done == 1:
                break

    print("***********************")
    print("Best ep{}".format(best_ep))
    print("Best reward", best_reward)
    print("Best Board:")
    print(best_board)
    print("Best step", best_steps)
    print("Best score", best_score)

    plot_info(
        steps_per_episode,
        rewards_per_episode,
        loss_per_episode,
        scores_per_episode,
        interval_mean,
        episodes,
        threshold,
    )


def main(screen):
    # Arguments
    args = parse_args()
    size_board = args.size_board
    episodes = args.num_episodes
    interval_mean = args.interval_mean
    hidden_dim = args.hidden_dim
    output_dim = 4

    # Initialize agent
    agent = Agent_Not_Train(
        size_board,
        hidden_dim,
        output_dim,
        path="./best_20000_not_change.pth"
    )

    # Initialize game
    env = Game2048Env(size_board)

    # Training
    play(
        env, agent, episodes, interval_mean, screen
    )


if __name__ == "__main__":
    grid_size = 70
    pygame.init()
    screen = pygame.display.set_mode((grid_size * 4, grid_size * 4), 0, 32)
    pygame.font.init()
    main(screen)
