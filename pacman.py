import copy
from board import boards
import pygame
import math
import matplotlib.pyplot as plt
from collections import deque
from agent import QLearningAgent
import os # We need this to set the environment variable

# This forces pygame to use a more compatible video driver.
# It often fixes windowing issues on some systems.
os.environ['SDL_VIDEODRIVER'] = 'windib'

# Basic setup for pygame and the game window.
pygame.init()
WIDTH = 900
HEIGHT = 950
screen = pygame.display.set_mode([WIDTH, HEIGHT])
game_surface = pygame.Surface((WIDTH, HEIGHT))

pygame.display.set_caption('Pac-Man RL Training')
timer = pygame.time.Clock()
fps = 120 # Faster fps for faster training.
font = pygame.font.Font('freesansbold.ttf', 20)
level = copy.deepcopy(boards)
color = 'blue'
PI = math.pi
Q_TABLE_FILE = 'q_table.pkl'

# Load all the game images.
player_images = []
for i in range(1, 5):
    player_images.append(pygame.transform.scale(pygame.image.load(f'assets/player_images/{i}.png'), (45, 45)))
blinky_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/red.png'), (45, 45))
pinky_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/pink.png'), (45, 45))
inky_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/blue.png'), (45, 45))
clyde_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/orange.png'), (45, 45))
spooked_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/powerup.png'), (45, 45))
dead_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/dead.png'), (45, 45))

# Game variables.
player_x = 450
player_y = 663
direction = 0
blinky_x = 56
blinky_y = 58
blinky_direction = 0
inky_x = 440
inky_y = 388
inky_direction = 2
pinky_x = 440
pinky_y = 438
pinky_direction = 2
clyde_x = 410
clyde_y = 438
clyde_direction = 2
counter = 0
flicker = False
turns_allowed = [False, False, False, False]
direction_command = 0
player_speed = 2
score = 0
powerup = False
power_counter = 0
eaten_ghost = [False, False, False, False]
targets = [(player_x, player_y), (player_x, player_y), (player_x, player_y), (player_x, player_y)]
blinky_dead = False
inky_dead = False
clyde_dead = False
pinky_dead = False
blinky_box = False
inky_box = False
clyde_box = False
pinky_box = False
moving = False
ghost_speeds = [2, 2, 2, 2]
startup_counter = 0
lives = 1
game_over = False
game_won = False

agent = QLearningAgent(actions=[0, 1, 2, 3])
agent.load_q_table(Q_TABLE_FILE)

TILE_WIDTH = WIDTH // 30
TILE_HEIGHT = (HEIGHT - 50) // 32

# Lists to track scores for the final graph.
scores = []
episodes = []
mean_scores = []
last_100_scores = deque(maxlen=100)

def plot_performance(episodes, scores, mean_scores):
    # This function creates and saves the performance graph.
    plt.figure()
    plt.title('Training Progress')
    plt.xlabel('Number of Games (Episodes)')
    plt.ylabel('Score')
    plt.plot(episodes, scores, label='Score per Episode')
    plt.plot(episodes, mean_scores, color='orange', linewidth=2, label='Mean Score (Last 100)')
    plt.legend()
    plt.ylim(ymin=0)
    plt.savefig('pacman_training_performance.png')
    plt.show()

def get_power_pellet_locations(current_level):
    # Finds the coordinates of all power pellets on the board.
    locations = []
    for i in range(len(current_level)):
        for j in range(len(current_level[i])):
            if current_level[i][j] == 2:
                locations.append((j, i))
    return tuple(sorted(locations))

def reset_game_state():
    # Resets the game to its starting state for a new episode.
    global player_x, player_y, direction, direction_command, score, powerup, power_counter, lives
    global blinky_x, blinky_y, blinky_direction, inky_x, inky_y, inky_direction
    global pinky_x, pinky_y, pinky_direction, clyde_x, clyde_y, clyde_direction
    global eaten_ghost, blinky_dead, inky_dead, clyde_dead, pinky_dead, level, game_over, game_won

    player_x, player_y, direction, direction_command = 450, 663, 0, 0
    score, lives, powerup, power_counter = 0, 1, False, 0
    blinky_x, blinky_y, blinky_direction = 56, 58, 0
    inky_x, inky_y, inky_direction = 440, 388, 2
    pinky_x, pinky_y, pinky_direction = 440, 438, 2
    clyde_x, clyde_y, clyde_direction = 410, 438, 2
    eaten_ghost = [False] * 4
    blinky_dead, inky_dead, clyde_dead, pinky_dead = False, False, False, False
    level = copy.deepcopy(boards)
    game_over, game_won = False, False
    
    ghost_coords = (
        (int(blinky_x // TILE_WIDTH), int(blinky_y // TILE_HEIGHT)),
        (int(inky_x // TILE_WIDTH), int(inky_y // TILE_HEIGHT)),
        (int(pinky_x // TILE_WIDTH), int(pinky_y // TILE_HEIGHT)),
        (int(clyde_x // TILE_WIDTH), int(clyde_y // TILE_HEIGHT)),
    )
    power_pellets = get_power_pellet_locations(level)
    
    return get_state(player_x, player_y, ghost_coords, power_pellets)

def get_state(p_x, p_y, ghost_coords, power_pellets):
    # Gathers all relevant game info to create the agent's current state.
    pac_x_grid = int((p_x + TILE_WIDTH / 2) // TILE_WIDTH)
    pac_y_grid = int((p_y + TILE_HEIGHT / 2) // TILE_HEIGHT)
    
    pac_y_grid = max(1, min(len(level) - 2, pac_y_grid))
    pac_x_grid = max(1, min(len(level[0]) - 2, pac_x_grid))

    # Check the four tiles around Pac-Man.
    tile_up = 1 if level[pac_y_grid - 1][pac_x_grid] < 3 else 0
    tile_down = 1 if level[pac_y_grid + 1][pac_x_grid] < 3 else 0
    tile_left = 1 if level[pac_y_grid][pac_x_grid - 1] < 3 else 0
    tile_right = 1 if level[pac_y_grid][pac_x_grid + 1] < 3 else 0
    adjacent_tiles = (tile_up, tile_down, tile_left, tile_right)

    # Get the positions of the ghosts relative to Pac-Man.
    ghost_positions = []
    for gx, gy in ghost_coords:
        rel_x = gx - pac_x_grid
        rel_y = gy - pac_y_grid
        ghost_positions.append((rel_x, rel_y))
    
    # The final state combines all this information.
    state = (
        adjacent_tiles,
        tuple(sorted(ghost_positions)),
        power_pellets
    )
    
    return state

class Ghost:
    def __init__(self, x_coord, y_coord, target, speed, img, direct, dead, box, id):
        self.x_pos = x_coord
        self.y_pos = y_coord
        self.center_x = self.x_pos + 22
        self.center_y = self.y_pos + 22
        self.target = target
        self.speed = speed
        self.img = img
        self.direction = direct
        self.dead = dead
        self.in_box = box
        self.id = id
        self.turns, self.in_box = self.check_collisions()
        self.rect = self.draw()

    def draw(self):
        # All drawing for the ghost is done on the game_surface.
        if (not powerup and not self.dead) or (eaten_ghost[self.id] and powerup and not self.dead):
            game_surface.blit(self.img, (self.x_pos, self.y_pos))
        elif powerup and not self.dead and not eaten_ghost[self.id]:
            game_surface.blit(spooked_img, (self.x_pos, self.y_pos))
        else:
            game_surface.blit(dead_img, (self.x_pos, self.y_pos))
        ghost_rect = pygame.rect.Rect((self.center_x - 18, self.center_y - 18), (36, 36))
        return ghost_rect

    def check_collisions(self):
        num1 = ((HEIGHT - 50) // 32); num2 = (WIDTH // 30); num3 = 15
        self.turns = [False, False, False, False]
        if 0 < self.center_x // 30 < 29:
            if level[(self.center_y - num3) // num1][self.center_x // num2] == 9: self.turns[2] = True
            if level[self.center_y // num1][(self.center_x - num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x - num3) // num2] == 9 and (self.in_box or self.dead)): self.turns[1] = True
            if level[self.center_y // num1][(self.center_x + num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x + num3) // num2] == 9 and (self.in_box or self.dead)): self.turns[0] = True
            if level[(self.center_y + num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y + num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[3] = True
            if level[(self.center_y - num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y - num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[2] = True
            if self.direction == 2 or self.direction == 3:
                if 12 <= self.center_x % num2 <= 18:
                    if level[(self.center_y + num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y + num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[3] = True
                    if level[(self.center_y - num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y - num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[2] = True
                if 12 <= self.center_y % num1 <= 18:
                    if level[self.center_y // num1][(self.center_x - num2) // num2] < 3 or (level[self.center_y // num1][(self.center_x - num2) // num2] == 9 and (self.in_box or self.dead)): self.turns[1] = True
                    if level[self.center_y // num1][(self.center_x + num2) // num2] < 3 or (level[self.center_y // num1][(self.center_x + num2) // num2] == 9 and (self.in_box or self.dead)): self.turns[0] = True
            if self.direction == 0 or self.direction == 1:
                if 12 <= self.center_x % num2 <= 18:
                    if level[(self.center_y + num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y + num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[3] = True
                    if level[(self.center_y - num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y - num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)): self.turns[2] = True
                if 12 <= self.center_y % num1 <= 18:
                    if level[self.center_y // num1][(self.center_x - num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x - num3) // num2] == 9 and (self.in_box or self.dead)): self.turns[1] = True
                    if level[self.center_y // num1][(self.center_x + num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x + num3) // num2] == 9 and (self.in_box or self.dead)): self.turns[0] = True
        else: self.turns[0] = True; self.turns[1] = True
        if 350 < self.x_pos < 550 and 370 < self.y_pos < 480: self.in_box = True
        else: self.in_box = False
        return self.turns, self.in_box

    def move_clyde(self):
        if self.direction == 0:
            if self.target[0] > self.x_pos and self.turns[0]: self.x_pos += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos += self.speed
        elif self.direction == 1:
            if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3
            elif self.target[0] < self.x_pos and self.turns[1]: self.x_pos -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos -= self.speed
        elif self.direction == 2:
            if self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y_pos and self.turns[3]: self.y_pos += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos += self.speed
        if self.x_pos < -30: self.x_pos = 900
        elif self.x_pos > 900: self.x_pos = -30
        return self.x_pos, self.y_pos, self.direction

    def move_blinky(self):
        if self.direction == 0:
            if self.target[0] > self.x_pos and self.turns[0]: self.x_pos += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[0]: self.x_pos += self.speed
        elif self.direction == 1:
            if self.target[0] < self.x_pos and self.turns[1]: self.x_pos -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[1]: self.x_pos -= self.speed
        elif self.direction == 2:
            if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[2]: self.y_pos -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y_pos and self.turns[3]: self.y_pos += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[3]: self.y_pos += self.speed
        if self.x_pos < -30: self.x_pos = 900
        elif self.x_pos > 900: self.x_pos = -30
        return self.x_pos, self.y_pos, self.direction

    def move_inky(self):
        if self.direction == 0:
            if self.target[0] > self.x_pos and self.turns[0]: self.x_pos += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos += self.speed
        elif self.direction == 1:
            if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3
            elif self.target[0] < self.x_pos and self.turns[1]: self.x_pos -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos -= self.speed
        elif self.direction == 2:
            if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[2]: self.y_pos -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y_pos and self.turns[3]: self.y_pos += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[3]: self.y_pos += self.speed
        if self.x_pos < -30: self.x_pos = 900
        elif self.x_pos > 900: self.x_pos = -30
        return self.x_pos, self.y_pos, self.direction

    def move_pinky(self):
        if self.direction == 0:
            if self.target[0] > self.x_pos and self.turns[0]: self.x_pos += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[0]: self.x_pos += self.speed
        elif self.direction == 1:
            if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3
            elif self.target[0] < self.x_pos and self.turns[1]: self.x_pos -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[1]: self.x_pos -= self.speed
        elif self.direction == 2:
            if self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y_pos and self.turns[3]: self.y_pos += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos += self.speed
        if self.x_pos < -30: self.x_pos = 900
        elif self.x_pos > 900: self.x_pos = -30
        return self.x_pos, self.y_pos, self.direction

def draw_misc():
    score_text = font.render(f'Score: {score}', True, 'white')
    game_surface.blit(score_text, (10, 920))
    if powerup: pygame.draw.circle(game_surface, 'blue', (140, 930), 15)
    for i in range(lives): game_surface.blit(pygame.transform.scale(player_images[0], (30, 30)), (650 + i * 40, 915))
    if game_over:
        pygame.draw.rect(game_surface, 'white', [50, 200, 800, 300],0, 10)
        pygame.draw.rect(game_surface, 'dark gray', [70, 220, 760, 260], 0, 10)
        gameover_text = font.render('Game over! Training episode ended.', True, 'red')
        game_surface.blit(gameover_text, (100, 300))

def check_collisions(scor, power, power_count, eaten_ghosts):
    num1 = (HEIGHT - 50) // 32; num2 = WIDTH // 30
    center_x = player_x + 22; center_y = player_y + 22
    if 0 < player_x < 870:
        if level[center_y // num1][center_x // num2] == 1:
            level[center_y // num1][center_x // num2] = 0; scor += 10
        if level[center_y // num1][center_x // num2] == 2:
            level[center_y // num1][center_x // num2] = 0; scor += 50
            power = True; power_count = 0; eaten_ghosts = [False] * 4
    return scor, power, power_count, eaten_ghosts

def draw_board():
    num1 = ((HEIGHT - 50) // 32); num2 = (WIDTH // 30)
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 1: pygame.draw.circle(game_surface, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 4)
            if level[i][j] == 2 and not flicker: pygame.draw.circle(game_surface, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 10)
            if level[i][j] == 3: pygame.draw.line(game_surface, color, (j * num2 + (0.5 * num2), i * num1), (j * num2 + (0.5 * num2), i * num1 + num1), 3)
            if level[i][j] == 4: pygame.draw.line(game_surface, color, (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)
            if level[i][j] == 5: pygame.draw.arc(game_surface, color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 + (0.5 * num1)), num2, num1], 0, PI / 2, 3)
            if level[i][j] == 6: pygame.draw.arc(game_surface, color, [(j * num2 + (num2 * 0.5)), (i * num1 + (0.5 * num1)), num2, num1], PI / 2, PI, 3)
            if level[i][j] == 7: pygame.draw.arc(game_surface, color, [(j * num2 + (num2 * 0.5)), (i * num1 - (0.4 * num1)), num2, num1], PI, 3 * PI / 2, 3)
            if level[i][j] == 8: pygame.draw.arc(game_surface, color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 - (0.4 * num1)), num2, num1], 3 * PI / 2, 2 * PI, 3)
            if level[i][j] == 9: pygame.draw.line(game_surface, 'white', (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)

def draw_player():
    if direction == 0: game_surface.blit(player_images[counter // 5], (player_x, player_y))
    elif direction == 1: game_surface.blit(pygame.transform.flip(player_images[counter // 5], True, False), (player_x, player_y))
    elif direction == 2: game_surface.blit(pygame.transform.rotate(player_images[counter // 5], 90), (player_x, player_y))
    elif direction == 3: game_surface.blit(pygame.transform.rotate(player_images[counter // 5], 270), (player_x, player_y))

def check_position(centerx, centery):
    turns = [False, False, False, False]
    num1 = (HEIGHT - 50) // 32; num2 = (WIDTH // 30); num3 = 15
    if centerx // 30 < 29:
        if direction == 0:
            if level[centery // num1][(centerx - num3) // num2] < 3: turns[1] = True
        if direction == 1:
            if level[centery // num1][(centerx + num3) // num2] < 3: turns[0] = True
        if direction == 2:
            if level[(centery + num3) // num1][centerx // num2] < 3: turns[3] = True
        if direction == 3:
            if level[(centery - num3) // num1][centerx // num2] < 3: turns[2] = True
        if direction == 2 or direction == 3:
            if 12 <= centerx % num2 <= 18:
                if level[(centery + num3) // num1][centerx // num2] < 3: turns[3] = True
                if level[(centery - num3) // num1][centerx // num2] < 3: turns[2] = True
            if 12 <= centery % num1 <= 18:
                if level[centery // num1][(centerx - num2) // num2] < 3: turns[1] = True
                if level[centery // num1][(centerx + num2) // num2] < 3: turns[0] = True
        if direction == 0 or direction == 1:
            if 12 <= centerx % num2 <= 18:
                if level[(centery + num3) // num1][centerx // num2] < 3: turns[3] = True
                if level[(centery - num3) // num1][centerx // num2] < 3: turns[2] = True
            if 12 <= centery % num1 <= 18:
                if level[centery // num1][(centerx - num3) // num2] < 3: turns[1] = True
                if level[centery // num1][(centerx + num3) // num2] < 3: turns[0] = True
    else: turns[0] = True; turns[1] = True
    return turns

def move_player(play_x, play_y):
    if direction == 0 and turns_allowed[0]: play_x += player_speed
    elif direction == 1 and turns_allowed[1]: play_x -= player_speed
    if direction == 2 and turns_allowed[2]: play_y -= player_speed
    elif direction == 3 and turns_allowed[3]: play_y += player_speed
    return play_x, play_y

# This is the main training loop.
run = True
total_episodes = 10000
for episode in range(total_episodes):
    state = reset_game_state()
    done = False
    step_counter = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                done = True
        
        timer.tick(fps)
        counter = (counter + 1) % 20
        flicker = (counter > 10)
        
        # All drawing is done on the off-screen surface.
        game_surface.fill('black')
        draw_board()
        draw_player()

        action = agent.choose_action(state)
        direction_command = action
        previous_score = score
        
        # Ghost logic.
        player_target = (player_x, player_y)
        blinky = Ghost(blinky_x, blinky_y, player_target, 2, blinky_img, blinky_direction, blinky_dead, blinky_box, 0)
        inky = Ghost(inky_x, inky_y, player_target, 2, inky_img, inky_direction, inky_dead, inky_box, 1)
        pinky = Ghost(pinky_x, pinky_y, player_target, 2, pinky_img, pinky_direction, pinky_dead, pinky_box, 2)
        clyde = Ghost(clyde_x, clyde_y, player_target, 2, clyde_img, clyde_direction, clyde_dead, clyde_box, 3)
        
        if inky.in_box: inky.target = (440, 300)
        if pinky.in_box: pinky.target = (440, 300)
        if clyde.in_box: clyde.target = (440, 300)
        
        blinky_x, blinky_y, blinky_direction = blinky.move_blinky()
        inky_x, inky_y, inky_direction = inky.move_inky()
        pinky_x, pinky_y, pinky_direction = pinky.move_pinky()
        clyde_x, clyde_y, clyde_direction = clyde.move_clyde()

        # Player logic.
        center_x = player_x + 22; center_y = player_y + 22
        turns_allowed = check_position(center_x, center_y)

        if direction_command == 0 and turns_allowed[0]: direction = 0
        if direction_command == 1 and turns_allowed[1]: direction = 1
        if direction_command == 2 and turns_allowed[2]: direction = 2
        if direction_command == 3 and turns_allowed[3]: direction = 3
        
        player_x, player_y = move_player(player_x, player_y)
        score, powerup, power_counter, eaten_ghost = check_collisions(score, powerup, power_counter, eaten_ghost)
        
        draw_misc()

        # Check for collisions and game over.
        player_circle = pygame.draw.circle(game_surface, 'black', (center_x, center_y), 20, 2)
        ghost_rects = [blinky.rect, inky.rect, pinky.rect, clyde.rect]
        for ghost_rect in ghost_rects:
            if player_circle.colliderect(ghost_rect) and not powerup:
                lives -= 1
                if lives <= 0: game_over = True
                break

        step_counter += 1
        if step_counter > 2000: game_over = True
        
        # Calculate reward and learn.
        reward = (score - previous_score) - 0.1
        if game_over:
            reward = -100
            done = True
        
        current_ghost_coords = (
            (int(blinky_x // TILE_WIDTH), int(blinky_y // TILE_HEIGHT)),
            (int(inky_x // TILE_WIDTH), int(inky_y // TILE_HEIGHT)),
            (int(pinky_x // TILE_WIDTH), int(pinky_y // TILE_HEIGHT)),
            (int(clyde_x // TILE_WIDTH), int(clyde_y // TILE_HEIGHT)),
        )
        current_power_pellets = get_power_pellet_locations(level)
        next_state = get_state(player_x, player_y, current_ghost_coords, current_power_pellets)
        
        agent.learn(state, action, reward, next_state)
        state = next_state
        
        # The entire game surface is drawn to the main screen at once.
        screen.blit(game_surface, (0, 0))
        pygame.display.flip()

    # After each episode, track the scores for the graph.
    scores.append(score)
    episodes.append(episode)
    last_100_scores.append(score)
    mean_score = sum(last_100_scores) / len(last_100_scores)
    mean_scores.append(mean_score)
    
    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    print(f"Ep {episode}: Score={score}, Mean Score={mean_score:.2f}, Epsilon={agent.epsilon:.4f}")

    if not run:
        break

# After all training is done.
print("Training complete. Saving Q-table and showing performance graph.")
agent.save_q_table(Q_TABLE_FILE)
plot_performance(episodes, scores, mean_scores)

pygame.quit()