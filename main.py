import pygame

from utils import parse_args
from agent import Agent
from env import Game2048Env
from train import train

grid_size = 70

def main(screen):
    # Arguments
    args = parse_args()
    seed = args.seed
    capacity = args.capacity
    size_board = args.size_board
    batch_size = args.batch_size
    episodes = args.num_episodes
    ep_update_target = args.ep_update_target
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    interval_mean = args.interval_mean
    gamma = args.gamma
    hidden_dim = args.hidden_dim
    dueling = args.dueling
    output_dim = 4

    # Initialize agent
    agent = Agent(
        size_board,
        hidden_dim,
        output_dim,
        decay_rate,
        capacity,
        batch_size,
        gamma,
        learning_rate,
        dueling,
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
        hidden_dim,
        screen
    )


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((grid_size * 4, grid_size * 4), 0, 32)
    pygame.font.init()
    main(screen)
