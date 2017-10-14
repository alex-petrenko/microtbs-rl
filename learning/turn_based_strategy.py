import os
import random
import logging

import pygame

import numpy as np

from utils import *


# pylint: disable=protected-access


logger = logging.getLogger(os.path.basename(__file__))


class Hero:
    def __init__(self, game):
        self.game = game
        self.pos = None
        self.color = (63, 63, 127)

    def draw(self):
        self.game._draw_tile(self.pos, self.color)

    def as_dict(self):
        return {
            'pos': self.pos,
        }


class Tile:
    ground, obstacle, gold = range(3)
    colors = {
        ground: (39, 40, 34),
        obstacle: (163, 163, 163),
        gold: (255, 191, 0),
    }

    @classmethod
    def color(cls, tile):
        return cls.colors[tile]


class Action:
    all_actions = range(9)
    noop, up, right, down, left, ul, ur, dl, dr = all_actions

    movement = {
        noop: (0, 0),
        up: (-1, 0),
        right: (0, 1),
        down: (1, 0),
        left: (0, -1),
        ul: (-1, -1),
        ur: (-1, 1),
        dl: (1, -1),
        dr: (1, 1),
    }

    @classmethod
    def delta(cls, action):
        return Vec(*cls.movement[action])

class Game:
    border = 1
    max_num_steps = 100

    def __init__(self, windowless=False, size=16, resolution=500):
        self.num_steps = 0
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

        self.screen = None
        if not windowless:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        self.clock = pygame.time.Clock()
        self.reset()

    def get_state(self):
        return {
            'world': self.world,
            'hero': self.hero.as_dict(),
        }

    def reset(self):
        self.num_steps = 0
        self.over = False

        dim = self.world_size
        world = np.full((dim, dim), Tile.ground)

        # setting world boundaries
        world[0] = world[dim - 1] = world[:, 0] = world[:, dim - 1] = Tile.obstacle

        def random_coord():
            return random.randrange(self.border, dim - self.border)
        def random_pos():
            return (random_coord(), random_coord())

        num_gold_piles = random.randint(1, 24)
        for _ in range(num_gold_piles):
            world[random_pos()] = Tile.gold

        self.hero.pos = None
        while self.hero.pos is None:
            pos = random_pos()
            if world[pos] == Tile.ground:
                self.hero.pos = Vec(*pos)

        self.world = world
        return self.get_state()

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
        self.num_steps += 1

        new_pos = self.hero.pos + Action.delta(action)
        reward = 0
        if self.world[new_pos.ij] == Tile.gold:
            reward += 0.1
            self.world[new_pos.ij] = Tile.ground
        elif self.world[new_pos.ij] != Tile.obstacle:
            self.hero.pos = new_pos

        gold_left = (self.world == Tile.gold).sum()
        if gold_left == 0:
            reward += 0.5
            self.over = True

        if self.num_steps > self.max_num_steps:
            self.over = True

        return self.get_state(), reward

    def render(self):
        self.screen.fill(Tile.color(Tile.ground))

        for (i, j), tile in np.ndenumerate(self.world):
            if tile == Tile.ground:
                continue
            self._draw_tile(Vec(i, j), Tile.color(tile))

        self.hero.draw()

        pygame.display.flip()

    def _draw_tile(self, pos, color):
        tsize = self.tile_size
        rect = pygame.Rect(pos.x * tsize, pos.y * tsize, tsize, tsize)
        pygame.draw.rect(self.screen, color, rect)
