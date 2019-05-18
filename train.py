import numpy as np
import torch
from copy import deepcopy

from frame_2048 import draw
from utils import plot_info


def train(
    episodes,
    agent,
    env,
    size_board,
    ep_update_target,
    interval_mean,
    dueling,
    batch_size,
    hidden_dim,
    screen
):

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

                if info["total_score"] > best_score:
                    best_score = info["total_score"]
                    best_reward = np.sum(episode_rewards)
                    best_ep = ep
                    best_board = deepcopy(next_state)
                    print(best_board)
                    best_steps = steps
                    agent.save_model("./best.pth")

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
    print("Hidden dim", hidden_dim)
    print("***********************")
    agent.save_model("./last.pth")

    plot_info(
        steps_per_episode,
        rewards_per_episode,
        loss_per_episode,
        scores_per_episode,
        interval_mean,
        episodes,
        threshold,
    )
