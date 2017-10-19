import os
import random
import logging

import pygame

import numpy as np

from utils import *


# pylint: disable=protected-access


logger = logging.getLogger(os.path.basename(__file__))


# Helper constants

DI = (-1, 0, 1, 0)
DJ = (0, 1, 0, -1)


# Various types used in the game

class Entity:
    @staticmethod
    def _color():
        """Default color, should be overridden."""
        return (255, 0, 255)

    def draw(self, game, surface, pos, scale):
        game._draw_tile(surface, pos, self._color(), scale)


class Terrain(Entity):
    @staticmethod
    def reachable():
        return True
    @staticmethod
    def penalty():
        return 1.0


class Ground(Terrain):
    @staticmethod
    def _color():
        return (39, 40, 34)


class Obstacle(Terrain):
    @staticmethod
    def reachable():
        return False

    @staticmethod
    def _color():
        return (73, 66, 58)


class Road(Terrain):
    @staticmethod
    def _color():
        return (174, 113, 92)

    @staticmethod
    def penalty():
        return 0.5


class Swamp(Terrain):
    @staticmethod
    def _color():
        return (54, 96, 58)

    @staticmethod
    def penalty():
        return 1.5


class GameObject(Entity):
    def __init__(self):
        self.disappear = False

    def interact(self, game):  # pylint: disable=no-self-use,unused-argument
        """Returns reward."""
        return 0

    @staticmethod
    def can_be_visited():
        """
        can_be_visited is True if we can "step" on the object, False otherwise.
        E.g. when we interact with a pile of gold, we can't step on it, we just collect it and
        then it disappears and the hero takes it's place.
        Staples or lookout towers, on the other hand, can be genuinely "visited".

        """
        return False

    def should_disappear(self):
        """
        A way to tell the game that this object should vanish during the current frame.
        E.g. a pile of gold that has just been visited.

        """
        return self.disappear

    @staticmethod
    def _color():
        """Default color, should be overridden."""
        return (255, 0, 255)


class GoldPile(GameObject):
    def __init__(self):
        super(GoldPile, self).__init__()
        min_size = 500
        step = 100
        size = random.randint(0, 5)
        self.amount = min_size + size * step

    def interact(self, game):
        game.hero.money += self.amount
        reward = self.amount / 10000.0
        game.num_gold_piles -= 1
        self.disappear = True
        return reward

    @staticmethod
    def _color():
        return (255, 191, 0)


class Hero(Entity):
    max_teams = 2
    teams = range(max_teams)
    team_red, team_blue = teams

    colors = {
        team_red: (255, 0, 0),
        team_blue: (0, 0, 255),
    }

    def __init__(self, start_movepoints, start_money=0, team=team_red):
        self.pos = None
        self.team = team
        self.movepoints = start_movepoints
        self.money = start_money

    def _color(self):
        return self.colors[self.team]

    def change_movepoints(self, delta):
        self.movepoints += delta
        self.movepoints = max(0, self.movepoints)


# class Tile:
#     # terrain
#     ground = 0
#     obstacle = 1
#     road = 2
#     swamp = 3

#     # resources
#     gold = 20

#     # objects
#     castle = 40
#     army_dwell = 41
#     lookout_tower = 42
#     stables = 43


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

    manhattan_to_euclid = {
        0: 0,
        1: 100,
        2: 141,
    }

    @staticmethod
    def delta(action):
        return Vec(*Action.movement[action])

    @staticmethod
    def movepoints(action, penalty):
        move_cells = sum((abs(d) for d in Action.movement[action]))
        return penalty * Action.manhattan_to_euclid[move_cells]


class Game:
    border = 1
    max_num_steps = 30

    def __init__(self, windowless=False, world_size=6, view_size=6, resolution=500):
        self.num_steps = 0
        self.over = False
        self.quit = False

        self.view_size = view_size
        dim = world_size + 2 * self.border
        self.world_size = dim
        self.terrain = None
        self.objects = None
        self.num_gold_piles = 0

        self.hero = None

        # pylint: disable=too-many-function-args
        self.state_surface = pygame.Surface((self.world_size, self.world_size))

        # reset the game world
        self.reset()

        # stuff related to game rendering
        self.screen_scale = 1
        while self.screen_scale * dim < resolution:
            self.screen_scale += 1

        self.screen_size = self.screen_scale * dim
        self.screen = None
        if not windowless:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()

    def reset(self):
        """Generate the new game."""
        self.num_steps = 0
        self.over = False
        self.hero = None
        dim = self.world_size

        while self.hero is None:
            self._generate_world()

            # place hero in the world
            unoccupied_cells = []
            for i in range(dim):
                for j in range(dim):
                    if self.objects[i, j] is not None:
                        continue
                    if not self.terrain[i, j].reachable():
                        continue
                    unoccupied_cells.append((i, j))
            if not unoccupied_cells:
                logger.info('World generation failed, try again...')
                continue

            start_movepoints = 100 * dim * dim / 2
            self.hero = Hero(team=Hero.team_red, start_movepoints=start_movepoints)
            hero_pos_idx = random.randrange(len(unoccupied_cells))
            self.hero.pos = Vec(*unoccupied_cells[hero_pos_idx])

        return self.get_state()

    def _generate_world(self):
        dim = self.world_size

        ground, obstacle, swamp = Ground(), Obstacle(), Swamp()
        self.terrain = np.full((dim, dim), ground, dtype=Terrain)

        # setting world boundaries
        self.terrain[0] = self.terrain[dim - 1] = obstacle
        self.terrain[:, 0] = self.terrain[:, dim - 1] = obstacle

        # generate game objects
        self.objects = np.full((dim, dim), None, dtype=GameObject)
        num_gold_piles = random.randint(1, int(dim * dim * 0.45))

        def random_coord():
            return random.randrange(self.border, dim - self.border)
        def random_pos():
            return (random_coord(), random_coord())

        for _ in range(num_gold_piles):
            self.objects[random_pos()] = GoldPile()

        obj_list = self.objects.flatten().tolist()
        self.num_gold_piles = sum((isinstance(obj, GoldPile) for obj in obj_list))

        # generate terrain
        self._generate_terrain((obstacle, swamp))

    def _generate_terrain(self, terrain):
        dim = self.world_size
        # generate terrain "seeds"
        seed_prob = 0.1
        for i in range(dim):
            for j in range(dim):
                if random.random() > seed_prob:
                    continue
                terrain_idx = random.randrange(len(terrain))
                self._spread_terrain(i, j, terrain[terrain_idx])

    def _spread_terrain(self, i, j, terrain):
        if self.objects[i, j] is not None:
            return
        if not isinstance(self.terrain[i, j], Ground):
            return
        self.terrain[i, j] = terrain
        spread_prob = 0.33
        for di, dj in zip(DI, DJ):
            if random.random() < spread_prob:
                self._spread_terrain(i + di, j + dj, terrain)

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

    def _win_condition(self):
        return self.num_gold_piles == 0

    def _lose_condition(self):
        if self.hero.movepoints == 0:
            return True
        if self.num_steps >= self.max_num_steps:
            return True
        return False

    def _game_over_condition(self):
        return self._win_condition() or self._lose_condition()

    def step(self, action):
        """Returns tuple (new_state, reward)."""
        self.num_steps += 1
        new_pos = self.hero.pos + Action.delta(action)
        reward = 0

        # required movepoints
        penalty = self.terrain[new_pos.ij].penalty()
        action_mp = Action.movepoints(action, penalty)
        self.hero.change_movepoints(-action_mp)

        can_move = True
        if self.hero.movepoints == 0:
            can_move = False
        else:
            obj = self.objects[new_pos.ij]
            if obj is not None:
                obj_reward = obj.interact(self)
                reward += obj_reward
                can_move = obj.can_be_visited()
                if obj.should_disappear():
                    self.objects[new_pos.ij] = None
                    del obj

            if not self.terrain[new_pos.ij].reachable():
                can_move = False

        if can_move:
            self.hero.pos = new_pos

        if self._win_condition():
            reward += 0.5

        if self._game_over_condition():
            self.over = True

        return self.get_state(), reward

    def _render_game_world(self, surface, scale):
        surface.fill((0, 0, 0))
        for (i, j), terrain in np.ndenumerate(self.terrain):
            terrain.draw(self, surface, Vec(i, j), scale)
        for (i, j), obj in np.ndenumerate(self.objects):
            if obj is not None:
                obj.draw(self, surface, Vec(i, j), scale)
        self.hero.draw(self, surface, self.hero.pos, scale)

    def _render_info(self):
        # TODO render info
        pass

    @staticmethod
    def _draw_tile(surface, pos, color, scale):
        rect = pygame.Rect(pos.x * scale, pos.y * scale, scale, scale)
        pygame.draw.rect(surface, color, rect)

    def _visual_state(self):
        self._render_game_world(self.state_surface, 1)
        view = pygame.surfarray.array3d(self.state_surface)
        view = view.astype(np.float32) / 255.0  # convert to format for DNN
        return view

    def _non_visual_state(self):
        return {
            'movepoints': self.hero.movepoints,
            'money': self.hero.money,
            'remaining_steps': self.max_num_steps - self.num_steps,
        }

    def get_state(self):
        return {
            'visual_state': self._visual_state(),
            'non_visual_state': self._non_visual_state(),
        }

    def render(self):
        self._render_game_world(self.screen, self.screen_scale)
        self._render_info()
        pygame.display.flip()
