import pygame

grid_size = 70


def draw(state, screen):
    # env.render()  # 刷新环境

    # Display with pygame
    black = (0, 0, 0)
    grey = (128, 128, 128)
    blue = (0, 0, 128)
    white = (255, 255, 255)
    tile_colour_map = {
        2: (255, 0, 0),
        4: (224, 32, 0),
        8: (192, 64, 0),
        16: (160, 96, 0),
        32: (128, 128, 0),
        64: (96, 160, 0),
        128: (64, 192, 0),
        256: (32, 224, 0),
        512: (0, 255, 0),
        1024: (0, 224, 32),
        2048: (0, 192, 64),
    }
    # Background
    screen.fill(black)
    # Board
    pygame.draw.rect(screen, grey, (0, 0, grid_size * 4, grid_size * 4))
    myfont = pygame.font.SysFont('Tahome', 30)
    for i, o, in enumerate(state):
        x = i % 4
        y = i // 4
        if o:
            pygame.draw.rect(screen, tile_colour_map[o],
                             (x * grid_size, y * grid_size, grid_size, grid_size))
            text = myfont.render(str(o), False, white)
            text_rect = text.get_rect()
            width = text_rect.width
            height = text_rect.height
            assert width < grid_size
            assert height < grid_size
            screen.blit(text, (x * grid_size + grid_size / 2 - text_rect.width / 2,
                               y * grid_size + grid_size / 2 - text_rect.height / 2))

    pygame.display.update()
