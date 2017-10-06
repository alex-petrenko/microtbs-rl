import os
import random
import logging

import pygame

import numpy as np

from utils import *


# pylint: disable=protected-access


logger = logging.getLogger(os.path.basename(__file__))


class Vec:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def x(self):
        return self.j

    @x.setter
    def x(self, x):
        self.j = x

    @property
    def y(self):
        return self.i

    @y.setter
    def y(self, y):
        self.i = y

    @property
    def ij(self):
        return (self.i, self.j)

    def __neg__(self):
        return Vec(-self.i, -self.j)

    def __add__(self, other):
        return Vec(self.i + other.i, self.j + other.j)

    def __sub__(self, other):
        return self + (-other)


class Hero:
    def __init__(self, game):
        self.game = game
        self.pos = None
        self.color = (63, 63, 127)

    def draw(self):
        self.game._draw_tile(self.pos, self.color)


class Tile:
    GROUND, OBSTACLE, GOLD = range(3)
    colors = {
        GROUND: (39, 40, 34),
        OBSTACLE: (163, 163, 163),
        GOLD: (255, 191, 0),
    }

    @classmethod
    def color(cls, tile):
        return cls.colors[tile]


class Action:
    all_actions = range(5)
    noop, up, right, down, left = all_actions

    movement = {
        noop: (0, 0),
        up: (-1, 0),
        right: (0, 1),
        down: (1, 0),
        left: (0, -1),
    }

    @classmethod
    def delta(cls, action):
        return Vec(*cls.movement[action])

class TurnBasedStrategy:
    border = 1

    def __init__(self, size=5, resolution=500):
        self.over = False
        self.quit = False

        dim = size + 2 * self.border
        self.world_size = dim
        self.world = np.ndarray(shape=(dim, dim), dtype=int)

        self.hero = Hero(game=self)

        self.tile_size = 1
        while self.tile_size * dim < resolution:
            self.tile_size += 1

        self.screen_size = self.tile_size * dim
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.over = False

        dim = self.world_size
        world = np.full((dim, dim), Tile.GROUND)

        # setting world boundaries
        world[0] = world[dim - 1] = world[:, 0] = world[:, dim - 1] = Tile.OBSTACLE

        def random_coord():
            return random.randrange(self.border, dim - self.border)
        def random_pos():
            return (random_coord(), random_coord())

        num_gold_piles = random.randint(1, 5)
        for _ in range(num_gold_piles):
            world[random_pos()] = Tile.GOLD

        self.hero.pos = None
        while self.hero.pos is None:
            pos = random_pos()
            if world[pos] == Tile.GROUND:
                self.hero.pos = Vec(*pos)

        self.world = world

    def is_over(self):
        return self.over

    def should_quit(self):
        return self.quit

    @staticmethod
    def allowed_actions():
        return Action.all_actions

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.over = self.quit = True
                return Action.noop

        a_map = {
            pygame.K_UP: Action.up,
            pygame.K_DOWN: Action.down,
            pygame.K_LEFT: Action.left,
            pygame.K_RIGHT: Action.right,
        }
        pressed = pygame.key.get_pressed()
        for key, action in a_map.items():
            if pressed[key]:
                return action
        return Action.noop

    def step(self, action):
        new_pos = self.hero.pos + Action.delta(action)
        reward = 0
        if self.world[new_pos.ij] == Tile.GOLD:
            reward += 0.01
            self.world[new_pos.ij] = Tile.GROUND
        elif self.world[new_pos.ij] != Tile.OBSTACLE:
            self.hero.pos = new_pos

        gold_left = (self.world == Tile.GOLD).sum()
        if gold_left == 0:
            reward += 0.5
            self.over = True

        return self.world, reward

    def render(self):
        self.screen.fill(Tile.color(Tile.GROUND))

        for (i, j), tile in np.ndenumerate(self.world):
            if tile == Tile.GROUND:
                continue
            self._draw_tile(Vec(i, j), Tile.color(tile))

        self.hero.draw()

        pygame.display.flip()

    def _draw_tile(self, pos, color):
        tsize = self.tile_size
        rect = pygame.Rect(pos.x * tsize, pos.y * tsize, tsize, tsize)
        pygame.draw.rect(self.screen, color, rect)
