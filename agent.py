import heapq
import random
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """
    Practical 02 - Step 1.2: Simple Reflex Agent
    Acts strictly on immediate sensory inputs (Condition-Action rules) with NO internal memory.
    """

    def sense_and_act(self, percept: dict) -> str:
        # Rule 1: IF standing on food, THEN stay
        if percept.get('food_here'):
            return 'Stay'

        # Rule 2: IF wall is ahead, THEN turn Right
        if percept.get('wall_ahead'):
            return 'Right'

        # Default Action: Move Up
        return 'Up'


class ModelBasedAgent:
    """
    Practical 02 - Step 1.3: Model-Based Agent
    Maintains internal state (memory of visited cells & position) to escape infinite loops.
    """

    def __init__(self):
        self.visited_cells = set()
        self.current_pos = [0, 0]
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        # 1. Transition Model: Update estimated internal position based on last action taken
        if self.last_action == 'Up':
            self.current_pos[1] += 1
        elif self.last_action == 'Right':
            self.current_pos[0] += 1
        elif self.last_action == 'Left':
            self.current_pos[0] -= 1
        elif self.last_action == 'Down':
            self.current_pos[1] -= 1

        # 2. Sensor Model: Record current estimated tile in visited history
        self.visited_cells.add(tuple(self.current_pos))

        # 3. Condition-Action Rules querying internal memory
        actions = ['Up', 'Right', 'Left', 'Down']

        if percept.get('wall_ahead'):
            action = random.choice(['Right', 'Left', 'Down'])
        else:
            action = 'Up' if tuple(self.current_pos) not in self.visited_cells else random.choice(actions)

        self.last_action = action
        return action


class SearchAgent:
    """
    Practical 03 - Goal-Based Planning Agent
    Uses BFS, DFS, or UCS graph search algorithms to plan paths to food.
    """

    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo  # Can be set to 'BFS', 'DFS', or 'UCS'

    def get_neighbors(self, pos, grid_size, walls):
        """Generates valid adjacent coordinate moves from the current position."""
        x, y = pos
        w, h = grid_size
        neighbors = []

        # Order: Up, Down, Left, Right
        if y + 1 < h and (x, y + 1) not in walls:
            neighbors.append(('Up', (x, y + 1)))
        if y - 1 >= 0 and (x, y - 1) not in walls:
            neighbors.append(('Down', (x, y - 1)))
        if x - 1 >= 0 and (x - 1, y) not in walls:
            neighbors.append(('Left', (x - 1, y)))
        if x + 1 < w and (x + 1, y) not in walls:
            neighbors.append(('Right', (x + 1, y)))

        return neighbors

    def bfs_search(self, start, goal, grid_size, walls):
        """Breadth-First Search using a FIFO Queue (deque.popleft)."""
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            current_pos, path = frontier.popleft()
            if current_pos == goal:
                return path

            for action, next_pos in self.get_neighbors(current_pos, grid_size, walls):
                if next_pos not in reached:
                    reached.add(next_pos)
                    frontier.append((next_pos, path + [action]))
        return ['Stay']

    def dfs_search(self, start, goal, grid_size, walls):
        """Depth-First Search using a LIFO Stack (list.pop)."""
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            current_pos, path = frontier.pop()
            if current_pos == goal:
                return path

            for action, next_pos in self.get_neighbors(current_pos, grid_size, walls):
                if next_pos not in reached:
                    reached.add(next_pos)
                    frontier.append((next_pos, path + [action]))
        return ['Stay']

    def ucs_search(self, start, goal, grid_size, walls):
        """Uniform-Cost Search using a Priority Queue ordered by path cost g(n)."""
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, start, []))
        reached = {start: 0}

        while frontier:
            cost, _, current_pos, path = heapq.heappop(frontier)
            if current_pos == goal:
                return path

            for action, next_pos in self.get_neighbors(current_pos, grid_size, walls):
                new_cost = cost + 1
                if next_pos not in reached or new_cost < reached[next_pos]:
                    reached[next_pos] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_pos, path + [action]))
        return ['Stay']

    def sense_and_act(self, percept: dict) -> str:
        # Step 1.3: Form a complete plan if current plan is empty and food remains
        if not self.plan and percept.get('all_food'):
            start = tuple(percept['agent_pos'])
            grid_size = percept['grid_size']
            walls = set(tuple(w) for w in percept['walls'])
            
            # Select the nearest food pellet based on Manhattan distance
            all_food = [tuple(f) for f in percept['all_food']]
            goal = min(all_food, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, grid_size, walls)

        # Step 1.3: Return the first action from the plan
        if self.plan:
            return self.plan.pop(0)

        return 'Stay'