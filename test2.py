import numpy as np
import pygame
from copy import deepcopy

import torch

from agent import Agent, Agent2
from env import Game2048Env
from frame_2048 import draw
from model import DDQN_MLP, DQN_MLP, Net1
from tensorboardX import SummaryWriter
from utils import plot_info, parse_args


def train(
    episodes,
    agent,
    env,
    size_board,
    ep_update_target,
    interval_mean,
    dueling,
    batch_size,
    hidden_dim_1,
    hidden_dim_2,
    hidden_dim_3,
    screen
):
    # 数据可视化
    writer = SummaryWriter(log_dir="data/test2")

    rewards_per_episode = []
    loss_per_episode = []
    steps_per_episode = []
    scores_per_episode = []
    threshold = []
    decay_step = 0
    best_score = 0

    for ep in range(episodes):
        print(ep)

        done = 0
        state, valid_movements = env.reset()
        loss_ep = []
        episode_rewards = []
        steps = 0

        while True:

            draw(state.flatten(), screen)
            steps += 1

            action = agent.selection_action(valid_movements, state.flatten())

            eps_threshold = agent.get_threshold()
            threshold.append(eps_threshold)

            next_state, reward, done, info = env.step(action)

            episode_rewards.append(reward)

            if done == 1:

                steps_per_episode.append(steps)

                rewards_per_episode.append(np.sum(episode_rewards))
                loss_per_episode.append(np.sum(loss_ep) / steps)
                scores_per_episode.append(info["total_score"])

                # loss可视化
                writer.add_scalar("data/test2/loss_groups", np.sum(loss_ep), ep)

                writer.add_scalar("data/test2/score_groups", reward, ep)

                if info["total_score"] > best_score:
                    best_score = info["total_score"]
                    best_reward = np.sum(episode_rewards)
                    best_ep = ep
                    best_board = deepcopy(next_state)
                    print(best_board)
                    best_steps = steps
                    agent.save_model("./test.pth")

                agent.store_memory(
                    state.flatten(), next_state.flatten(), action, reward, done
                )

                next_state = np.zeros((1, size_board * size_board))
            else:
                # print(next_state)
                agent.store_memory(
                    state.flatten(), next_state.flatten(), action, reward, done
                )

                state = deepcopy(next_state)

                valid_movements = info["valid_movements"]

            loss = agent.train_model()

            if loss != -1:

                loss_ep.append(loss)

            if done == 1:
                break

        if ep % ep_update_target == 0:
            print("Update")
            agent.update_target_net()

    device = torch.device("cuda")
    batch_state = torch.tensor(state.flatten(), dtype=torch.float32).view(1, 1, -1)

    with SummaryWriter(log_dir="data/test2", comment='DQN_NET') as w:

        w.add_graph(Net1(size_board * size_board, hidden_dim_1, hidden_dim_2, hidden_dim_3, 4), (batch_state,))
    writer.export_scalars_to_json("data/all_scalars.json")
    writer.close()

    print("***********************")
    print("Best ep", best_ep)
    print("Best Board:")
    print(best_board)
    print("Best step", best_steps)
    print("Best score", best_score)
    if dueling is True:
        print("Dueling type")
    else:
        print("No-dueling type")
    print("Update Target_Net period", ep_update_target)
    print("Batch size", batch_size)
    print("***********************")
    agent.save_model("./test.pth")

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
    capacity = args.capacity
    size_board = args.size_board
    batch_size = args.batch_size
    episodes = args.num_episodes
    ep_update_target = args.ep_update_target
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    interval_mean = args.interval_mean
    gamma = args.gamma

    hidden_dim_1 = args.hidden_dim_1
    hidden_dim_2 = args.hidden_dim_2
    hidden_dim_3 = args.hidden_dim_3

    dueling = args.dueling
    output_dim = 4

    # Initialize agent
    agent = Agent2(
        size_board,
        hidden_dim_1,
        hidden_dim_2,
        hidden_dim_3,
        output_dim,
        decay_rate,
        capacity,
        batch_size,
        gamma,
        learning_rate,
        False
    )

    # Initialize game
    env = Game2048Env(size_board)

    # Training
    train(
        episodes,
        agent,
        env,
        size_board,
        ep_update_target,
        interval_mean,
        dueling,
        batch_size,
        hidden_dim_1,
        hidden_dim_2,
        hidden_dim_3,
        screen
    )


if __name__ == "__main__":
    grid_size = 70
    pygame.init()
    screen = pygame.display.set_mode((grid_size * 4, grid_size * 4), 0, 32)
    pygame.font.init()
    main(screen)
