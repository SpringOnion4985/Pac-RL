import copy
from board import boards
import pygame
import math
# import matplotlib.pyplot as plt # FIX: Removed for now to simplify
from collections import deque
from agent import QLearningAgent

# Basic setup for pygame and the game window.
pygame.init()
WIDTH = 900
HEIGHT = 950
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Pac-Man RL Training')
timer = pygame.time.Clock()
fps = 120 # Faster fps for faster training.
font = pygame.font.Font('freesansbold.ttf', 20)
level = copy.deepcopy(boards)
color = 'blue'
PI = math.pi

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

# Original game variables
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
## FIX: Changed Clyde's starting x-position so he doesn't overlap with Pinky.
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
TILE_WIDTH = WIDTH // 30
TILE_HEIGHT = (HEIGHT - 50) // 32

def reset_game_state():
    global player_x, player_y, direction, direction_command, score, powerup, power_counter, lives
    global blinky_x, blinky_y, blinky_direction, inky_x, inky_y, inky_direction
    global pinky_x, pinky_y, pinky_direction, clyde_x, clyde_y, clyde_direction
    global eaten_ghost, blinky_dead, inky_dead, clyde_dead, pinky_dead, level, game_over, game_won

    player_x, player_y, direction, direction_command = 450, 663, 0, 0
    score, lives, powerup, power_counter = 0, 1, False, 0
    blinky_x, blinky_y, blinky_direction = 56, 58, 0
    inky_x, inky_y, inky_direction = 440, 388, 2
    pinky_x, pinky_y, pinky_direction = 440, 438, 2
    clyde_x, clyde_y, clyde_direction = 410, 438, 2 # ## FIX: Using the corrected coordinate here too.
    eaten_ghost = [False] * 4
    blinky_dead, inky_dead, clyde_dead, pinky_dead = False, False, False, False
    level = copy.deepcopy(boards)
    game_over, game_won = False, False
    return get_state(player_x, player_y)

def get_state(p_x, p_y):
    grid_x = int((p_x + TILE_WIDTH / 2) // TILE_WIDTH)
    grid_y = int((p_y + TILE_HEIGHT / 2) // TILE_HEIGHT)
    return (grid_x, grid_y)

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
        if (not powerup and not self.dead) or (eaten_ghost[self.id] and powerup and not self.dead):
            screen.blit(self.img, (self.x_pos, self.y_pos))
        elif powerup and not self.dead and not eaten_ghost[self.id]:
            screen.blit(spooked_img, (self.x_pos, self.y_pos))
        else:
            screen.blit(dead_img, (self.x_pos, self.y_pos))
        ghost_rect = pygame.rect.Rect((self.center_x - 18, self.center_y - 18), (36, 36))
        return ghost_rect

    def check_collisions(self):
        num1 = ((HEIGHT - 50) // 32)
        num2 = (WIDTH // 30)
        num3 = 15
        self.turns = [False, False, False, False]
        if 0 < self.center_x // 30 < 29:
            if level[(self.center_y - num3) // num1][self.center_x // num2] == 9:
                self.turns[2] = True
            if level[self.center_y // num1][(self.center_x - num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x - num3) // num2] == 9 and (self.in_box or self.dead)):
                self.turns[1] = True
            if level[self.center_y // num1][(self.center_x + num3) // num2] < 3 or (level[self.center_y // num1][(self.center_x + num3) // num2] == 9 and (self.in_box or self.dead)):
                self.turns[0] = True
            if level[(self.center_y + num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y + num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)):
                self.turns[3] = True
            if level[(self.center_y - num3) // num1][self.center_x // num2] < 3 or (level[(self.center_y - num3) // num1][self.center_x // num2] == 9 and (self.in_box or self.dead)):
                self.turns[2] = True
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
        else:
            self.turns[0] = True
            self.turns[1] = True
        if 350 < self.x_pos < 550 and 370 < self.y_pos < 480:
            self.in_box = True
        else:
            self.in_box = False
        return self.turns, self.in_box
    
    # --- All original move functions are below, with one bug fix for the portal logic ---

    def move_clyde(self):
        # Your original move_clyde logic
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
        elif self.x_pos > 900: self.x_pos = -30 ## FIX: Was 'self.x_pos - 30'
        return self.x_pos, self.y_pos, self.direction

    def move_blinky(self):
        # Your original move_blinky logic
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
        elif self.x_pos > 900: self.x_pos = -30 ## FIX: Was 'self.x_pos -= 30'
        return self.x_pos, self.y_pos, self.direction

    def move_inky(self):
        # Your original move_inky logic
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
        elif self.x_pos > 900: self.x_pos = -30 ## FIX: Was 'self.x_pos -= 30'
        return self.x_pos, self.y_pos, self.direction

    def move_pinky(self):
        # Your original move_pinky logic
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
        elif self.x_pos > 900: self.x_pos = -30 ## FIX: Was 'self.x_pos -= 30'
        return self.x_pos, self.y_pos, self.direction

# --- UNCHANGED FUNCTIONS ---
# All drawing functions, player movement functions, and collision checks are the same as your original code.
# I am re-pasting them here for completeness.
def draw_misc():
    score_text = font.render(f'Score: {score}', True, 'white')
    screen.blit(score_text, (10, 920))
    if powerup:
        pygame.draw.circle(screen, 'blue', (140, 930), 15)
    for i in range(lives):
        screen.blit(pygame.transform.scale(player_images[0], (30, 30)), (650 + i * 40, 915))
    if game_over:
        pygame.draw.rect(screen, 'white', [50, 200, 800, 300],0, 10)
        pygame.draw.rect(screen, 'dark gray', [70, 220, 760, 260], 0, 10)
        gameover_text = font.render('Game over! Training episode ended.', True, 'red')
        screen.blit(gameover_text, (100, 300))
def check_collisions(scor, power, power_count, eaten_ghosts):
    num1 = (HEIGHT - 50) // 32
    num2 = WIDTH // 30
    center_x = player_x + 22
    center_y = player_y + 22
    if 0 < player_x < 870:
        if level[center_y // num1][center_x // num2] == 1:
            level[center_y // num1][center_x // num2] = 0; scor += 10
        if level[center_y // num1][center_x // num2] == 2:
            level[center_y // num1][center_x // num2] = 0; scor += 50
            power = True; power_count = 0; eaten_ghosts = [False] * 4
    return scor, power, power_count, eaten_ghosts
def draw_board():
    num1 = ((HEIGHT - 50) // 32)
    num2 = (WIDTH // 30)
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 1: pygame.draw.circle(screen, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 4)
            if level[i][j] == 2 and not flicker: pygame.draw.circle(screen, 'white', (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)), 10)
            if level[i][j] == 3: pygame.draw.line(screen, color, (j * num2 + (0.5 * num2), i * num1), (j * num2 + (0.5 * num2), i * num1 + num1), 3)
            if level[i][j] == 4: pygame.draw.line(screen, color, (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)
            if level[i][j] == 5: pygame.draw.arc(screen, color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 + (0.5 * num1)), num2, num1], 0, PI / 2, 3)
            if level[i][j] == 6: pygame.draw.arc(screen, color, [(j * num2 + (num2 * 0.5)), (i * num1 + (0.5 * num1)), num2, num1], PI / 2, PI, 3)
            if level[i][j] == 7: pygame.draw.arc(screen, color, [(j * num2 + (num2 * 0.5)), (i * num1 - (0.4 * num1)), num2, num1], PI, 3 * PI / 2, 3)
            if level[i][j] == 8: pygame.draw.arc(screen, color, [(j * num2 - (num2 * 0.4)) - 2, (i * num1 - (0.4 * num1)), num2, num1], 3 * PI / 2, 2 * PI, 3)
            if level[i][j] == 9: pygame.draw.line(screen, 'white', (j * num2, i * num1 + (0.5 * num1)), (j * num2 + num2, i * num1 + (0.5 * num1)), 3)
def draw_player():
    if direction == 0: screen.blit(player_images[counter // 5], (player_x, player_y))
    elif direction == 1: screen.blit(pygame.transform.flip(player_images[counter // 5], True, False), (player_x, player_y))
    elif direction == 2: screen.blit(pygame.transform.rotate(player_images[counter // 5], 90), (player_x, player_y))
    elif direction == 3: screen.blit(pygame.transform.rotate(player_images[counter // 5], 270), (player_x, player_y))
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
# --- END OF UNCHANGED FUNCTIONS ---


# --- MAIN TRAINING LOOP ---
run = True
total_episodes = 5000
for episode in range(total_episodes):
    state = reset_game_state()
    done = False
    episode_score = 0
    step_counter = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                done = True
        
        timer.tick(fps)
        counter = (counter + 1) % 20
        flicker = (counter > 10)
        
        screen.fill('black')
        draw_board()
        draw_player()

        action = agent.choose_action(state)
        direction_command = action
        previous_score = score
        
        # --- GHOST LOGIC FIXES ---
        
        # Create ghost objects
        player_target = (player_x, player_y)
        blinky = Ghost(blinky_x, blinky_y, player_target, 2, blinky_img, blinky_direction, blinky_dead, blinky_box, 0)
        inky = Ghost(inky_x, inky_y, player_target, 2, inky_img, inky_direction, inky_dead, inky_box, 1)
        pinky = Ghost(pinky_x, pinky_y, player_target, 2, pinky_img, pinky_direction, pinky_dead, pinky_box, 2)
        clyde = Ghost(clyde_x, clyde_y, player_target, 2, clyde_img, clyde_direction, clyde_dead, clyde_box, 3)
        
        ## FIX: Give ghosts a target to escape the box if they are inside it.
        if inky.in_box: inky.target = (440, 338)
        if pinky.in_box: pinky.target = (440, 338)
        if clyde.in_box: clyde.target = (440, 338)
        
        ## FIX: Call the move function for ALL ghosts and update their positions.
        blinky_x, blinky_y, blinky_direction = blinky.move_blinky()
        inky_x, inky_y, inky_direction = inky.move_inky()
        pinky_x, pinky_y, pinky_direction = pinky.move_pinky()
        clyde_x, clyde_y, clyde_direction = clyde.move_clyde()

        # --- PLAYER LOGIC ---
        center_x = player_x + 22
        center_y = player_y + 22
        turns_allowed = check_position(center_x, center_y)

        if direction_command == 0 and turns_allowed[0]: direction = 0
        if direction_command == 1 and turns_allowed[1]: direction = 1
        if direction_command == 2 and turns_allowed[2]: direction = 2
        if direction_command == 3 and turns_allowed[3]: direction = 3
        
        player_x, player_y = move_player(player_x, player_y)
        score, powerup, power_counter, eaten_ghost = check_collisions(score, powerup, power_counter, eaten_ghost)
        
        draw_misc()

        ## FIX: Check for collision with ALL four ghosts.
        player_circle = pygame.draw.circle(screen, 'black', (center_x, center_y), 20, 2)
        ghost_rects = [blinky.rect, inky.rect, pinky.rect, clyde.rect]
        for ghost_rect in ghost_rects:
            if player_circle.colliderect(ghost_rect) and not powerup:
                lives -= 1
                if lives <= 0:
                    game_over = True
                break # Exit loop once a collision is found

        step_counter += 1
        if step_counter > 2000:
            game_over = True
        
        # --- RL AGENT LOGIC ---
        reward = (score - previous_score) - 0.1
        if game_over:
            reward = -100
            done = True
        
        next_state = get_state(player_x, player_y)
        agent.learn(state, action, reward, next_state)
        state = next_state
        episode_score = score # Simplified score tracking

        pygame.display.flip()

    # After an episode ends.
    print(f"Episode {episode}: Score = {episode_score}")

    if not run:
        break

pygame.quit()