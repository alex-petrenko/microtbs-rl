import heapq
import random

from collections import deque

import numpy as np
import pygame

from utils import *

logger = logging.getLogger(os.path.basename(__file__))

# Helper constants

DI = (-1, 0, 1, 0)
DJ = (0, 1, 0, -1)


# Various types used in the game

class Entity:
    """Basically, anything within the game world."""

    def _color(self):
        """Default color, should be overridden."""
        return 255, 0, 255

    def draw(self, game, surface, pos, scale):
        game.draw_tile(surface, pos, self._color(), scale)


class Terrain(Entity):
    @staticmethod
    def reachable():
        return True

    @staticmethod
    def penalty():
        return 1.0


class Ground(Terrain):
    def _color(self):
        return 39, 40, 34


class Obstacle(Terrain):
    @staticmethod
    def reachable():
        return False

    def _color(self):
        return 73, 66, 58


class Swamp(Terrain):
    def _color(self):
        return 56, 73, 5

    @staticmethod
    def penalty():
        return 1.5


class GameObject(Entity):
    def __init__(self):
        self.disappear = False

    def interact(self, game, hero):
        """Returns reward."""
        return 0

    @staticmethod
    def can_be_visited():
        """
        can_be_visited is True if we can "step" on the object, False otherwise.
        E.g. when we interact with a pile of gold, we can't step on it, we just collect it and
        then it disappears and the hero takes it's place.
        Stables or lookout towers, on the other hand, can be genuinely "visited".

        """
        return False

    def should_disappear(self):
        """
        A way to tell the game that this object should vanish during the current frame.
        E.g. a pile of gold that has just been visited.

        """
        return self.disappear

    def on_new_day(self):
        """Called when game day changes."""
        pass

    def _color(self):
        """Default color, should be overridden."""
        return 255, 0, 255


class GoldPile(GameObject):
    def __init__(self):
        super(GoldPile, self).__init__()
        min_size = 500
        step = 100
        size = random.randint(0, 5)
        self.amount = min_size + size * step

    def interact(self, game, hero):
        hero.money += self.amount
        reward = (self.amount / 10.0) * game.reward_unit
        game.num_gold_piles -= 1
        self.disappear = True
        return reward

    def _color(self):
        return 255, 191, 0


class Stables(GameObject):
    def __init__(self):
        super(Stables, self).__init__()
        self.cost = 500
        self.movepoints = 750
        self.visited = {}

    def interact(self, game, hero):
        if self._can_be_visited_by_hero(hero):
            hero.money -= self.cost
            hero.change_movepoints(delta=self.movepoints)
            self.visited[hero.team] = True
            return game.reward_unit * 50
        return 0

    def _can_be_visited_by_hero(self, hero):
        visited = self.visited.get(hero.team, False)
        return not visited and hero.money >= self.cost

    @staticmethod
    def can_be_visited():
        return True

    def on_new_day(self):
        self.visited = {}  # can be visited once a day

    def _color(self):
        return 128, 64, 10

    def draw(self, game, surface, pos, scale):
        """Draw a tile a little bit dimmer if it cannot be visited."""
        color = self._color()
        if self._can_be_visited_by_hero(game.current_hero()):
            color = tuple(int(1.3 * c) for c in color)
        game.draw_tile(surface, pos, color, scale)


class LookoutTower(GameObject):
    def __init__(self):
        super(LookoutTower, self).__init__()
        self.cost = 100
        self.visited = {}

    def interact(self, game, hero):
        if self._can_be_visited_by_hero(hero):
            hero.money -= self.cost
            game.update_scouting(hero, scouting=(hero.scouting * 3))
            self.visited[hero.team] = True
            return game.reward_unit * 50
        return 0

    def _can_be_visited_by_hero(self, hero):
        visited = self.visited.get(hero.team, False)
        return not visited and hero.money >= self.cost

    @staticmethod
    def can_be_visited():
        return True

    def _color(self):
        return 57, 120, 140

    def draw(self, game, surface, pos, scale):
        """Draw a tile a little bit dimmer if it cannot be visited."""
        color = self._color()
        if self._can_be_visited_by_hero(game.current_hero()):
            color = tuple(int(1.3 * c) for c in color)
        game.draw_tile(surface, pos, color, scale)


class ArmyDwelling(GameObject):
    def __init__(self):
        super(ArmyDwelling, self).__init__()
        self.cost_per_unit = 1000
        self.num_units_per_day = 2
        self.num_units = self.num_units_per_day

    def interact(self, game, hero):
        if self._can_be_visited_by_hero(hero):
            hero.money -= self.cost_per_unit
            assert self.num_units > 0
            self.num_units -= 1
            hero.army += 1
            return game.reward_unit * 50
        return 0

    def _can_be_visited_by_hero(self, hero):
        return hero.money >= self.cost_per_unit and self.num_units > 0

    @staticmethod
    def can_be_visited():
        return True

    def on_new_day(self):
        """Number of units in the dwelling is restored every day."""
        self.num_units = self.num_units_per_day

    def _color(self):
        return 190, 190, 190

    def draw(self, game, surface, pos, scale):
        """Draw a tile a little bit dimmer if it cannot be visited."""
        color = self._color()
        if self._can_be_visited_by_hero(game.current_hero()):
            color = tuple(int(1.3 * c) for c in color)
        game.draw_tile(surface, pos, color, scale)


class Hero(Entity):
    max_teams = 2
    teams = range(max_teams)
    team_red, team_blue = teams

    colors = {
        team_red: (255, 0, 0),
        team_blue: (0, 0, 255),
    }

    def __init__(self, game, start_movepoints, start_money=0, team=team_red):
        self.finished_turn = False  # turn is over for this hero
        self.finished = False  # is game over for this hero
        self.pos = None
        self.team = team
        self.start_movepoints = start_movepoints
        self.movepoints = start_movepoints
        self.money = start_money
        self.income = 300  # small income every day
        self.army = 1  # start with a single unit in the army
        self.scouting = 13.25

        # initially, everything hero sees is fog of war
        dim = game.world_size
        self.fog_of_war = np.full((dim, dim), 1, dtype=np.uint8)

    def _color(self):
        return self.colors[self.team]

    def change_movepoints(self, delta):
        self.movepoints += delta
        self.movepoints = max(0, self.movepoints)

    def reset_day(self):
        self.money += self.income
        self.finished_turn = False
        self.movepoints = self.start_movepoints

    def within_scouting_range(self, i, j, scouting):
        return self.pos.dist_sq(Vec(i, j)) <= (scouting ** 2)

    def is_defeated(self):
        return self.army <= 0

    @staticmethod
    def heroes_interaction(hero_active, hero_passive):
        assert hero_active.team != hero_passive.team
        logger.info('Heroes interact! %d vs %d', hero_active.team, hero_passive.team)

        def transfer_money(hero_from, hero_to):
            hero_to.money += hero_from.money
            hero_from.money = 0

        if hero_active.army > hero_passive.army:
            transfer_money(hero_passive, hero_active)
        elif hero_active.army < hero_passive.army:
            transfer_money(hero_active, hero_passive)

        army_min = min(hero_active.army, hero_passive.army)
        hero_active.army -= army_min
        hero_passive.army -= army_min


class Action:
    all_actions = range(9)
    up, right, down, left, ul, ur, dl, dr, noop = all_actions

    movement = {
        up: (-1, 0),
        right: (0, 1),
        down: (1, 0),
        left: (0, -1),
        ul: (-1, -1),
        ur: (-1, 1),
        dl: (1, -1),
        dr: (1, 1),
        noop: (0, 0),
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
    def movepoints(action, penalty_coeff):
        move_cells = sum((abs(d) for d in Action.movement[action]))
        return penalty_coeff * Action.manhattan_to_euclid[move_cells]


class WorldOptions:
    def __init__(self):
        self.diagonal_moves = False
        self.num_teams = 1
        self.max_days = 3
        self.check_endgame = None

        # world generation options
        self.terrain_seed_prob = {Obstacle: 0.07, Swamp: 0.06}

    @staticmethod
    def collect_gold_simple():
        opt = WorldOptions()
        opt.check_endgame = WorldOptions._check_endgame_collect
        opt.max_days = 1
        return opt

    @staticmethod
    def pvp():
        opt = WorldOptions()
        opt.num_teams = 2
        opt.max_days = 3
        opt.check_endgame = WorldOptions._check_endgame_pvp
        return opt

    @staticmethod
    def _check_endgame_collect(game, *_):
        if game.num_gold_piles == 0:
            # collected all gold piles
            return True, +game.reward_unit * 1000
        return False, 0

    @staticmethod
    def _check_endgame_pvp(game, hero):
        if hero.is_defeated():
            return True, -game.reward_unit * 1000

        num_defeated = sum([h.is_defeated() for h in game.heroes])
        if num_defeated == len(game.heroes):
            # stalemate, all heroes are defeated
            return True, -game.reward_unit

        if not hero.is_defeated() and num_defeated == (len(game.heroes) - 1):
            # everyone is defeated except us, we won!
            return True, +game.reward_unit * 1000

        return False, 0


class Game:
    reward_unit = 0.001

    def __init__(self, world_options=None, windowless=False, world_size=5, view_size=7, resolution=600):
        self.world_opt = WorldOptions() if world_options is None else world_options

        self.over = False
        self.quit = False
        self.key_down = False
        self.num_steps = 0
        self.day = -1

        self.view_size = view_size
        self.camera_pos = Vec(0, 0)

        self.border = (self.view_size // 2) + 1
        self.world_size = world_size + 2 * self.border
        self.terrain = None
        self.objects = None
        self.num_gold_piles = 0

        self.heroes = []
        self.hero_idx, self.start_hero_idx = -1, -1

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
        self.over = False
        self.key_down = False
        self.num_steps = 0
        self.day = 1

        self.heroes = []
        # decide at random who's turn is first
        self.start_hero_idx = random.randrange(0, self.world_opt.num_teams)
        self.hero_idx = self.start_hero_idx

        while len(self.heroes) < self.world_opt.num_teams:
            self._generate_world()
            if self.num_gold_piles < 1:
                logger.info('World with zero gold generated, try again!')
                continue
            self._try_place_heroes()

        self.update_scouting(self.current_hero())

        # setup camera
        camera_i = max(0, self.current_hero().pos.i - self.view_size // 2)
        camera_j = max(0, self.current_hero().pos.j - self.view_size // 2)
        self.camera_pos = Vec(camera_i, camera_j)
        self._update_camera_position()

        return self.get_state()

    def _try_place_heroes(self):
        dim = self.world_size

        # Find all accessible space in the world, open areas where game objects are found. Otherwise heroes may
        # stuck, completely surronded by obstacles.
        accessible = np.full((dim, dim), False, dtype=bool)
        q = deque([])

        # all cells that are accessible, but not occupied by any object
        unoccupied_cells = []

        # we know that all objects are accessible, let's find the object and start search from there
        for (i, j), obj in np.ndenumerate(self.objects):
            if obj is not None:
                accessible[i, j] = True
                q.append((i, j))
                break

        # bfs
        while q:
            i, j = q.popleft()
            for di, dj in zip(DI, DJ):
                new_i, new_j = i + di, j + dj
                if not self.terrain[new_i, new_j].reachable():
                    continue
                if accessible[new_i, new_j]:
                    continue
                accessible[new_i, new_j] = True
                q.append((new_i, new_j))
                if self.objects[new_i, new_j] is None:
                    unoccupied_cells.append((new_i, new_j))

        if len(unoccupied_cells) < self.world_opt.num_teams:
            logger.info('World generation failed, try again...')
            return

        # equal amount of movepoints for all heroes
        min_movepoints = 2000
        max_movepoints = max(min_movepoints, 250 * dim * 2)
        max_movepoints = min(max_movepoints, 2500)
        start_movepoints = random.randint(min_movepoints, max_movepoints)

        # place heroes at the random locations in the world
        for team_idx in range(self.world_opt.num_teams):
            team = Hero.teams[team_idx]
            hero = Hero(self, team=team, start_movepoints=start_movepoints)
            hero_pos_idx = random.randrange(len(unoccupied_cells))
            hero.pos = Vec(*unoccupied_cells[hero_pos_idx])
            unoccupied_cells.pop(hero_pos_idx)  # do not place other heroes in the same tile
            self.heroes.append(hero)

    def _generate_world(self):
        dim = self.world_size

        # reset counters
        self.num_gold_piles = 0

        ground, obstacle, swamp = Ground(), Obstacle(), Swamp()
        # noinspection PyTypeChecker
        self.terrain = np.full((dim, dim), ground, dtype=Terrain)

        # setting world boundaries
        for i in range(self.border):
            self.terrain[i] = self.terrain[dim - i - 1] = obstacle
            self.terrain[:, i] = self.terrain[:, dim - i - 1] = obstacle

        # generate game objects
        self._generate_objects()

        # generate terrain
        terrains = (obstacle, swamp)
        self._generate_terrain(ground, terrains)

    def _generate_objects(self):
        dim = self.world_size
        # noinspection PyTypeChecker
        self.objects = np.full((dim, dim), None, dtype=GameObject)

        min_max_count_per_100_cells = {
            GoldPile: (3, 20),
            Stables: (0.5, 1),
            LookoutTower: (0.66, 1),
            ArmyDwelling: (0, 0),
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

    def _generate_terrain(self, default_terrain, terrains):
        """
        Generate underlying terrain.
        :param terrains: list of tuples (terrain, probability_of_terrain_seed)
        """
        dim = self.world_size
        # generate terrain "seeds"
        for i in range(dim):
            for j in range(dim):
                terrain = random.choice(terrains)
                seed_prob = self.world_opt.terrain_seed_prob[type(terrain)]
                if random.random() > seed_prob:
                    continue
                self._spread_terrain(i, j, terrain)

        self._ensure_all_objects_are_accessible(default_terrain)

    def _spread_terrain(self, seed_i, seed_j, terrain):
        """Use bfs instead of dfs to avoid possibly deep recursion."""
        def should_spread(cell_i, cell_j):
            if self.objects[cell_i, cell_j] is not None:
                return False
            return isinstance(self.terrain[cell_i, cell_j], Ground)

        if not should_spread(seed_i, seed_j):
            return
        q = deque([(seed_i, seed_j)])
        self.terrain[seed_i, seed_j] = terrain

        spread_prob = 0.4
        while q:
            i, j = q.popleft()
            for di, dj in zip(DI, DJ):
                new_i, new_j = i + di, j + dj
                if not should_spread(new_i, new_j):
                    continue
                self.terrain[new_i, new_j] = terrain
                if random.random() < spread_prob:
                    q.append((new_i, new_j))

    def _is_border(self, i, j):
        border = lambda coord: coord < self.border or coord >= self.world_size - self.border
        return border(i) or border(j)

    def _mark_zone(self, zones, start_i, start_j, zone_idx):
        """Run a bfs until all cells accessible from this one are marked with the same zone idx."""
        q = deque([(start_i, start_j)])
        zones[start_i, start_j] = zone_idx
        while q:
            i, j = q.popleft()
            for di, dj in zip(DI, DJ):
                new_i, new_j = i + di, j + dj
                if zones[new_i, new_j] != -1:
                    continue
                if self.terrain[new_i, new_j].reachable():
                    zones[new_i, new_j] = zone_idx
                    q.append((new_i, new_j))

    def _ensure_all_objects_are_accessible(self, default_terrain):
        """
        After generation of obstacles some areas of the map may be inaccessible.
        The following code removes some obstacles to make it possible for heroes to reach all objects in the world.
        :param default_terrain: replace obstacles with this type of terrain where needed.
        """
        dim = self.world_size
        zones = np.full((dim, dim), -1, dtype=int)
        num_zones = 0

        # find connected components of the world
        for (i, j), obj in np.ndenumerate(self.objects):
            if obj is not None and zones[i, j] == -1:
                self._mark_zone(zones, i, j, num_zones)
                num_zones += 1

        if num_zones <= 0:
            logger.info('World with no objects, skip...')
            return

        # Select one connected component and find shortest path to all other components.
        # Using Dijkstra algorithm to find shortest route, obstacle cells are given very high weight.
        selected_zone = random.randrange(0, num_zones)
        search_start = None
        for (i, j), zone in np.ndenumerate(zones):
            if zone == selected_zone:
                search_start = (i, j)
                break

        # Simplest implementation of the Dijkstra algorithm
        dist = np.full((dim, dim), 1000 * 1000 * 1000, dtype=int)
        dist[search_start] = 0

        search_buffer = []
        heapq.heappush(search_buffer, (0, search_start))

        visited = set()

        # noinspection PyTypeChecker
        path = np.full((dim, dim), None, dtype=Vec)

        while True:
            # Find not yet visited cell with shortest distance from the start.
            try:
                best_d, best_cell = heapq.heappop(search_buffer)
            except IndexError:
                break

            if best_cell in visited:
                continue
            visited.add(best_cell)

            # do edge relaxation
            best_cell = Vec(*best_cell)
            for di, dj in zip(DI, DJ):
                new_i, new_j = best_cell.i + di, best_cell.j + dj
                if self._is_border(new_i, new_j):
                    continue

                terrain = self.terrain[new_i, new_j]
                d = terrain.penalty()
                if not terrain.reachable():
                    d *= 10000

                new_dist = d + dist[best_cell.ij]
                if new_dist < dist[new_i, new_j]:
                    heapq.heappush(search_buffer, (new_dist, (new_i, new_j)))
                    dist[new_i, new_j] = new_dist
                    path[new_i, new_j] = best_cell

        zone_reachable = {selected_zone}
        for (i, j), zone in np.ndenumerate(zones):
            if zone == -1 or zone in zone_reachable:
                continue
            # recover shortest path to the point inside zone and remove all obstacles on this path
            cell_on_path = Vec(i, j)
            while cell_on_path is not None:
                if not self.terrain[cell_on_path.ij].reachable():
                    self.terrain[cell_on_path.ij] = default_terrain
                cell_on_path = path[cell_on_path.ij]
            zone_reachable.add(zone)

    def is_over(self):
        return self.over

    def should_quit(self):
        return self.quit

    def allowed_actions(self):
        if self.world_opt.diagonal_moves:
            # remove noop for the RL agents, keep it only for human version
            actions = list(Action.all_actions)
            actions.remove(Action.noop)
        else:
            actions = [Action.up, Action.right, Action.down, Action.left]
        return actions

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

    def _next_turn(self):
        reward = 0
        self.hero_idx = (self.hero_idx + 1) % self.world_opt.num_teams
        if self.hero_idx == self.start_hero_idx:
            # all heroes finished their moves, go to the next game day
            self._next_day()
            reward -= self.reward_unit * 10  # additional penalty for every day it took to finish the game
        return self.heroes[self.hero_idx], reward

    def _next_day(self):
        self.day += 1
        # give heroes some movepoints for the next day
        for hero in self.heroes:
            hero.reset_day()
        for i in range(self.world_size):
            for j in range(self.world_size):
                obj = self.objects[i, j]
                if obj is not None:
                    obj.on_new_day()  # some objects may change state between days

    def _game_over_condition(self):
        if self.day > self.world_opt.max_days:
            return True
        heroes_finished = sum([h.finished for h in self.heroes])
        return heroes_finished == len(self.heroes)  # all heroes are in finished state

    def current_hero(self):
        return self.heroes[self.hero_idx]

    def _game_step(self, action, hero):
        reward = 0
        new_pos = hero.pos + Action.delta(action)

        # required movepoints
        penalty_coeff = self.terrain[new_pos.ij].penalty()
        reward -= penalty_coeff * self.reward_unit  # tiny penalty for every step taken
        action_mp = Action.movepoints(action, penalty_coeff)

        can_move = True
        if hero.movepoints < action_mp:
            hero.movepoints = 0
            can_move = False
        else:
            hero.change_movepoints(delta=-action_mp)

            # first of all, check interactions with other heroes
            for i, other_hero in enumerate(self.heroes):
                if i == self.hero_idx:  # heroes should not interact with themselves
                    continue
                if other_hero.pos != new_pos:
                    continue
                Hero.heroes_interaction(hero_active=hero, hero_passive=other_hero)
                can_move = False

            # if no hero interaction, check if we can interact with a game object in the new_pos
            if can_move:
                obj = self.objects[new_pos.ij]
                if obj is not None:
                    obj_reward = obj.interact(self, hero)
                    reward += obj_reward
                    can_move = obj.can_be_visited()
                    if obj.should_disappear():
                        self.objects[new_pos.ij] = None
                        del obj

            # no heroes or objects, check if we bumped into an obstacle
            if can_move:
                if not self.terrain[new_pos.ij].reachable():
                    can_move = False
                    # small penalty for bumping into obstacles
                    reward -= 50 * self.reward_unit

        if can_move:
            hero.pos = new_pos

        # Check if we have enough movepoints to make at least one more move. If no, then its the end of the turn.
        min_next_mp = sys.maxsize
        for next_action in self.allowed_actions():
            next_pos = hero.pos + Action.delta(next_action)
            min_next_mp = min(min_next_mp, Action.movepoints(next_action, self.terrain[next_pos.ij].penalty()))
        if min_next_mp > hero.movepoints:
            # force the end of the turn
            hero.movepoints = 0

        hero_done, endgame_reward = self.world_opt.check_endgame(self, hero)
        if hero_done:
            hero.movepoints = 0
            hero.finished = True
            reward += endgame_reward

        if hero.movepoints == 0:
            hero.finished_turn = True

        return reward

    def step(self, action):
        """Returns tuple (new_state, reward)."""
        self.num_steps += 1
        hero = self.current_hero()

        if hero.finished_turn:
            # Here we have an "empty" step, where action taken by the hero does not matter.
            # We use it just to pass the control to the next hero.
            hero, reward = self._next_turn()
        else:
            # Regular update of the world dynamics.
            reward = self._game_step(action, hero)

        self.update_scouting(hero)
        self._update_camera_position()

        if self._game_over_condition():
            self.over = True

        return self.get_state(), reward

    def update_scouting(self, hero, scouting=None):
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
                if hero.fog_of_war[i, j] > 0 and hero.within_scouting_range(i, j, scouting):
                    hero.fog_of_war[i, j] = 0

    def _update_camera_position(self):
        # hero = self.current_hero()

        # this just centers camera at the world center, useful for small worlds
        self.camera_pos = Vec(
            self.world_size // 2 - self.view_size // 2,
            self.world_size // 2 - self.view_size // 2,
        )

        # move camera with the hero, keeping minimal clearance to the borders of the screen
        # def update_coord(pos, hero_pos):
        #     min_clearance = self.view_size // 2 - 1
        #     pos = min(pos, hero_pos - min_clearance)
        #     pos = max(pos, hero_pos - self.view_size + 1 + min_clearance)
        #     return pos
        #
        # self.camera_pos.i = update_coord(self.camera_pos.i, hero.pos.i)
        # self.camera_pos.j = update_coord(self.camera_pos.j, hero.pos.j)

    def _render_game_world(self, surface, scale):
        camera = self.camera_pos
        assert camera.i >= 0 and camera.j >= 0
        min_i, min_j = camera.i, camera.j
        max_i = min(self.world_size, min_i + self.view_size)
        max_j = min(self.world_size, min_j + self.view_size)

        surface.fill((9, 5, 6))
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                if self.current_hero().fog_of_war[i, j] > 0:
                    continue

                self.terrain[i, j].draw(self, surface, Vec(i, j) - camera, scale)
                obj = self.objects[i, j]
                if obj is not None:
                    obj.draw(self, surface, Vec(i, j) - camera, scale)

        # current hero should always be within the camera view
        assert min_i <= self.current_hero().pos.i < max_i
        assert min_j <= self.current_hero().pos.j < max_j

        for hero in self.heroes:
            pos = hero.pos - camera
            if 0 <= pos.i < self.view_size and 0 <= pos.j < self.view_size:
                if self.current_hero().fog_of_war[hero.pos.ij] == 0:
                    hero.draw(self, surface, pos, scale)

    def _render_info(self):
        hero = self.current_hero()
        self.ui_surface.fill((0, 0, 0))
        text_items = [
            'Team: ' + str(self.current_hero().team),
            'Movepoints: ' + str(hero.movepoints),
            'Gold: ' + str(hero.money),
            'Army: ' + str(hero.army),
            'Day: ' + str(self.day),
            'Finished: ' + str(hero.finished),
            'Win: ' + str(hero.finished and not hero.is_defeated()),
        ]
        offset = self.screen.get_height() // 50
        x_offset = y_offset = offset
        text_color = (248, 248, 242)
        for h in self.heroes:
            if h.finished and not h.is_defeated():
                text_color = Hero.colors[h.team]
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
    def draw_tile(surface, pos, color, scale):
        rect = pygame.Rect(pos.x * scale, pos.y * scale, scale, scale)
        pygame.draw.rect(surface, color, rect)

    def _visual_state(self):
        self._render_game_world(self.state_surface, 1)
        view = pygame.surfarray.array3d(self.state_surface)
        view = view.astype(np.float32) / 255.0  # convert to format for DNN
        return view

    def _non_visual_state(self):
        hero = self.current_hero()
        return {
            'team': hero.team,
            'movepoints': hero.movepoints,
            'money': hero.money,
            'army': hero.army,
            'day': self.day,
            'finished': int(hero.finished),
            'win': int(hero.finished and not hero.is_defeated()),
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
