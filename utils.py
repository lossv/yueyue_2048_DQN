import numpy as np
import argparse
import matplotlib.pyplot as plt


# 参数修改
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=100000)  # 记忆容量
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size_board", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=30000)  # 运行轮数
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--ep_update_target", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.00005)
    parser.add_argument("--interval_mean", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dueling", default=False, action="store_true")

    # Net1 参数
    parser.add_argument("--hidden_dim_1", type=int, default=64)
    parser.add_argument("--hidden_dim_2", type=int, default=128)
    parser.add_argument("--hidden_dim_3", type=int, default=32)

    args = parser.parse_args()

    return args


def get_mean_interval(array, interval_mean):
    interval_mean_list = []
    for x in range(interval_mean):
        interval_mean_list.append(0)

    for i in range(len(array)):
        if i + interval_mean == len(array):
            break
        else:
            interval_mean_list.append(np.mean(array[i: interval_mean + i]))

    return interval_mean_list


def plot_info(
        total_steps_per_episode,
        total_rewards_per_episode,
        total_loss_per_episode,
        total_score_per_episode,
        interval_mean,
        episodes,
        threshold,
):
    interval_steps = get_mean_interval(total_steps_per_episode, interval_mean)
    plt.plot(range(episodes), total_steps_per_episode)
    plt.plot(range(episodes), interval_steps)
    plt.ylabel("Episode durations")
    plt.xlabel("Episodes")
    plt.savefig("images/episodes_durations.png", bbox_inches="tight")
    plt.close()

    interval_rewards = get_mean_interval(total_rewards_per_episode, interval_mean)
    plt.plot(range(episodes), total_rewards_per_episode)
    plt.plot(range(episodes), interval_rewards)
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    plt.savefig("images/episodes_rewards.png", bbox_inches="tight")
    plt.close()

    interval_score = get_mean_interval(total_score_per_episode, interval_mean)
    plt.plot(range(episodes), total_score_per_episode)
    plt.plot(range(episodes), interval_score)
    plt.ylabel("Score")
    plt.xlabel("Episodes")
    plt.savefig("images/episodes_scores.png", bbox_inches="tight")
    plt.close()

    interval_loss = get_mean_interval(total_loss_per_episode, interval_mean)
    plt.plot(range(episodes), total_loss_per_episode)
    plt.plot(range(episodes), interval_loss)
    plt.ylabel("Loss")
    plt.xlabel("Episodes")
    plt.savefig("images/episodes_losses.png", bbox_inches="tight")
    plt.close()

    plt.plot(range(len(threshold)), threshold)
    plt.ylabel("Epsilon_threshold")
    plt.xlabel("Steps")
    plt.savefig("images/threshold.png", bbox_inches="tight")
    plt.close()
