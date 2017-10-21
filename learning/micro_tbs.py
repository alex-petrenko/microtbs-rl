import os
import math
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
        return (67, 70, 40)

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


class Stables(GameObject):
    def __init__(self):
        super(Stables, self).__init__()
        self.cost = 500
        self.movepoints = 500
        self.visited = {}

    def interact(self, game):
        visited = self.visited.get(game.hero.team, False)
        if not visited and game.hero.money >= self.cost:
            game.hero.money -= self.cost
            game.hero.change_movepoints(self.movepoints)
            self.visited[game.hero.team] = True
        return 0

    @staticmethod
    def can_be_visited():
        return True

    @staticmethod
    def _color():
        return (111, 84, 55)


class LookoutTower(GameObject):
    def __init__(self):
        super(LookoutTower, self).__init__()
        self.cost = 1000
        self.visited = {}

    def interact(self, game):
        visited = self.visited.get(game.hero.team, False)
        if not visited and game.hero.money >= self.cost:
            game.hero.money -= self.cost
            game._update_scouting(scouting=(game.hero.scouting * 3))
            self.visited[game.hero.team] = True
        return 0

    @staticmethod
    def can_be_visited():
        return True

    @staticmethod
    def _color():
        return (57, 120, 140)


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
        self.scouting = 3.5

    def _color(self):
        return self.colors[self.team]

    def change_movepoints(self, delta):
        self.movepoints += delta
        self.movepoints = max(0, self.movepoints)

    def within_scouting_range(self, i, j, scouting):
        return self.pos.dist_sq(Vec(i, j)) <= (scouting ** 2)


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
    max_num_steps = 100

    def __init__(self, windowless=False, world_size=30, view_size=12, resolution=700):
        self.num_steps = 0
        self.over = False
        self.quit = False
        self.key_down = False

        self.view_size = view_size
        self.camera_pos = Vec(0, 0)

        self.border = (self.view_size // 2) + 1
        self.world_size = world_size + 2 * self.border
        self.terrain = None
        self.objects = None
        self.fog_of_war = None
        self.num_gold_piles = 0

        self.hero = None

        # pylint: disable=too-many-function-args
        self.state_surface = pygame.Surface((self.view_size, self.view_size))

        # reset the game world
        self.reset()

        # stuff related to game rendering
        self.screen_scale = 1
        while self.screen_scale * self.view_size < resolution:
            self.screen_scale += 1

        self.screen = None
        if not windowless:
            self.screen_size_world = self.screen_scale * self.view_size
            ui_size = min(200, resolution // 3)
            screen_w = self.screen_size_world + ui_size
            screen_h = self.screen_size_world
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            self.font = pygame.font.SysFont(None, resolution // 30)
            self.ui_surface = pygame.Surface((ui_size, screen_h))
        self.clock = pygame.time.Clock()

    def reset(self):
        """Generate the new game."""
        self.num_steps = 0
        self.over = False
        self.key_down = False

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

            start_movepoints = 100 * dim * dim / 4
            self.hero = Hero(team=Hero.team_red, start_movepoints=start_movepoints)
            hero_pos_idx = random.randrange(len(unoccupied_cells))
            self.hero.pos = Vec(*unoccupied_cells[hero_pos_idx])

        self._update_scouting()

        # setup camera
        camera_i = max(0, self.hero.pos.i - self.view_size // 2)
        camera_j = max(0, self.hero.pos.j - self.view_size // 2)
        self.camera_pos = Vec(camera_i, camera_j)
        self._update_camera_position()

        return self.get_state()

    def _generate_world(self):
        dim = self.world_size

        # initially, everything is covered by fog of war
        self.fog_of_war = np.full((dim, dim), 1, dtype=np.uint8)

        ground, obstacle, swamp = Ground(), Obstacle(), Swamp()
        self.terrain = np.full((dim, dim), ground, dtype=Terrain)

        # setting world boundaries
        for i in range(self.border):
            self.terrain[i] = self.terrain[dim - i - 1] = obstacle
            self.terrain[:, i] = self.terrain[:, dim - i - 1] = obstacle

        # generate game objects
        self._generate_objects()

        # generate terrain
        self._generate_terrain((obstacle, swamp))

    def _generate_objects(self):
        dim = self.world_size
        self.objects = np.full((dim, dim), None, dtype=GameObject)

        min_max_count_per_100_cells = {
            GoldPile: (1, 33),
            Stables: (0.5, 1),
            LookoutTower: (0.33, 1),
        }

        probability = {}
        for obj_type, count_range in min_max_count_per_100_cells.items():
            approx_count_per_100 = random.uniform(*count_range)
            probability[obj_type] = approx_count_per_100 / 100.0

        obj_types = list(min_max_count_per_100_cells.keys())
        np.random.shuffle(obj_types)
        for i in range(dim):
            for j in range(dim):
                if not isinstance(self.terrain[i, j], Ground):
                    continue
                for obj_type in obj_types:
                    if random.random() < probability[obj_type]:
                        self.objects[i, j] = obj_type()
                        if obj_type == GoldPile:
                            self.num_gold_piles += 1
                        break

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
        events = []
        for event in pygame.event.get():
            events.append(event.type)
        if pygame.QUIT in events:
            self.over = self.quit = True
            return Action.noop

        selected_action = Action.noop
        if not self.key_down:
            a_map = {
                pygame.K_UP: Action.up,
                pygame.K_DOWN: Action.down,
                pygame.K_LEFT: Action.left,
                pygame.K_RIGHT: Action.right,
            }
            pressed = pygame.key.get_pressed()
            for key, action in a_map.items():
                if pressed[key]:
                    self.key_down = True
                    selected_action = action
                    break

        if pygame.KEYUP in events:
            self.key_down = False

        return selected_action

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

        self._update_scouting()

        self._update_camera_position()

        return self.get_state(), reward

    def _update_scouting(self, scouting=None):
        hero = self.hero
        if scouting is None:
            scouting = hero.scouting

        def approx_scouting_range(hero_coord):
            _min = max(0, hero_coord - math.ceil(scouting))
            _max = min(self.world_size, hero_coord + math.ceil(scouting) + 1)
            return range(_min, _max)
        range_i = approx_scouting_range(hero.pos.i)
        range_j = approx_scouting_range(hero.pos.j)

        for i in range_i:
            for j in range_j:
                if self.fog_of_war[i, j] > 0 and hero.within_scouting_range(i, j, scouting):
                    self.fog_of_war[i, j] = 0

    def _update_camera_position(self):
        def update_coord(pos, hero_pos):
            min_offset = (self.view_size // 2) - 1
            pos = min(pos, hero_pos - min_offset)
            pos = max(pos, hero_pos + min_offset - self.view_size)
            pos = min(pos, self.world_size - self.view_size)
            pos = max(pos, 0)
            return pos
        self.camera_pos.i = update_coord(self.camera_pos.i, self.hero.pos.i)
        self.camera_pos.j = update_coord(self.camera_pos.j, self.hero.pos.j)

    def _render_game_world(self, surface, scale):
        camera = self.camera_pos
        assert camera.i >= 0 and camera.j >= 0
        min_i, min_j = camera.i, camera.j
        max_i = min(self.world_size, min_i + self.view_size)
        max_j = min(self.world_size, min_j + self.view_size)

        surface.fill((9, 5, 6))
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                if self.fog_of_war[i, j] > 0:
                    continue

                self.terrain[i, j].draw(self, surface, Vec(i, j) - camera, scale)
                obj = self.objects[i, j]
                if obj is not None:
                    obj.draw(self, surface, Vec(i, j) - camera, scale)

        assert self.hero.pos.i >= min_i and self.hero.pos.i < max_i
        assert self.hero.pos.j >= min_j and self.hero.pos.j < max_j
        self.hero.draw(self, surface, self.hero.pos - camera, scale)

    def _render_info(self):
        self.ui_surface.fill((0, 0, 0))
        text_color = (248, 248, 242)
        text_items = [
            'Movepoints: ' + str(self.hero.movepoints),
            'Gold: ' + str(self.hero.money),
            'Time left: ' + str(self.max_num_steps - self.num_steps),
        ]
        offset = self.screen.get_height() // 50
        x_offset = y_offset = offset
        for text in text_items:
            label = self.font.render(text, 1, text_color)
            self.ui_surface.blit(label, (x_offset, y_offset))
            y_offset += label.get_height() + offset
        self.screen.blit(self.ui_surface, (self.screen_size_world, 0))

    def _render_layout(self):
        col = (24, 131, 215)
        w, h = self.screen_size_world, self.screen_size_world
        pygame.draw.line(self.screen, col, (w, 0), (w, h), 2)

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
        self._render_layout()
        pygame.display.flip()
