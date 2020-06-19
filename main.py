import argparse
import time
from os import path, walk
import numpy as np

states_idx = dict()
states_name = []
grid_idx = []
actions_name = {'move-east': 0,
                'move-north': 1,
                'move-west': 2,
                'move-south': 3}

actions = {-1: 0,
           0: '>',
           1: '^',
           2: '<',
           3: 'v'}

input_dir = path.realpath('TestGrid')


class MDP:

    def __init__(self, num_states=0, epsilon=0.1):
        self.V = self.next_V = None
        self.epsilon = epsilon
        self.iter = self.initial_state = self.goal = self.run_time = -1

        if num_states != 0:
            self.set_state_num(num_states)

        else:
            self.states = self.size = num_states
            self.transitions = self.costs = None

        self.initial_policy = self.policy = None

    def set_state_num(self, num_states):
        self.states = num_states
        self.transitions = [[[] for _ in range(4)] for _ in range(self.states)]
        self.costs = np.zeros((self.states, 4))
        self.size = max(get_state_pos(states_name[-1])) + 1
        # self.costs = [[1.0 for _ in range(4)] for _ in range(self.states)]

    def add_action(self, source: int, destination: int, action_idx: int, probability: float):
        self.transitions[source][action_idx].append((destination, probability))

    def add_cost(self, source: int, action_idx: int, cost: float):
        self.costs[source][action_idx] = cost

    def set_goal(self, new_goal: int):
        self.goal: int = new_goal

    def set_initial_state(self, new_initial_state: int):
        self.initial_state = new_initial_state

    def _bellman_backup(self, state: int):
        if state == self.goal:
            return -1, 0

        Q_s = np.full(4, np.nan)

        for action in range(4):
            if self.transitions[state][action]:
                # Q_s[action] = self.costs[state][action]
                # for successor, probability in self.transitions[state][action]:
                #     Q_s[action] += self.V[successor] * probability
                Q_s[action] = self.costs[state][action] + sum(self.V[succ] * prob
                                                              for succ, prob in self.transitions[state][action])

        return int(np.argmin(Q_s)), float(np.min(Q_s))

    def _init_var(self, policy=False):
        self.V = np.full(self.states, 30)
        self.V[self.goal] = 0
        self.next_V = np.zeros(self.states)
        self.iter = 0

        if policy:
            # self.policy = np.array([self._get_proper_action(state) for state in range(self.states)], dtype=int)
            self.policy = self._make_proper_policy()
            # self.policy[self.goal] = -1

            # self.initial_policy = deepcopy(self.policy)
            self.initial_policy = self.policy.copy()

    def value_iteration(self):
        start_time = time.time()
        self._init_var()
        max_res = self.epsilon

        # TODO: Implement bar logging
        while max_res >= self.epsilon:
            # print(self.iter, max_res)
            max_res = 0
            # self.aux_V = self.V.copy()
            self.iter += 1

            for state in range(self.states):
                self.next_V[state] = self._bellman_backup(state)[1]
                max_res = max(max_res, abs(self.next_V[state] - self.V[state]))

            # max_res = np.max(np.abs(new_V-self.V))
            # self.V = self.aux_V
            self.V = self.next_V.copy()
            # self.V, self.aux_V = self.aux_V, self.V

        self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]
        self.run_time = time.time() - start_time
        return self.policy

    def _make_proper_policy(self):
        # states_grid_idx = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        # for state in range(self.states):
        #     x, y = get_state_pos(states_name[state])
        #     states_grid_idx[x][y] = state

        def get_predecessors(state_idx):
            x, y = get_state_pos(states_name[state_idx])
            n1 = [x - 1, x + 1, x, x]
            n2 = [y, y, y - 1, y + 1]
            neighbours = []

            for i in range(len(n1)):
                if 0 <= n1[i] < self.size and 0 <= n2[i] < self.size and grid_idx[n1[i]][n2[i]] != -1:
                    # n_idx = states_grid_idx[n1[i]][n2[i]]
                    # if n_idx != -1 and policy[n_idx] == -1:
                    neighbours.append(grid_idx[n1[i]][n2[i]])

            return neighbours

        policy = np.full(self.states, -1, dtype=int)
        policy[self.goal] = 0

        define_predecessors = [self.goal]
        while define_predecessors:
            state = define_predecessors.pop()
            for predecessor in get_predecessors(state):
                if policy[predecessor] == -1:
                    policy[predecessor] = self._get_proper_action(predecessor, state)
                    define_predecessors.append(predecessor)

        policy[self.goal] = -1
        return policy

    def _get_proper_action(self, state, successor):
        for idx, action in enumerate(self.transitions[state]):
            if len(action) >= 1:
                for transition in action:
                    if transition[0] == successor:
                        # if successors[0] != state:
                        return idx
        return 0

    def policy_iteration(self):
        start_time = time.time()
        self._init_var(True)
        old_policy = np.full(self.states, -1, dtype=int)

        # for state in range(self.states):
        #     self.policy[state] = self._bellman_backup(state)[0]

        while not np.array_equal(old_policy, self.policy):
            self.iter += 1
            # print(self.iter, np.sum(old_policy == self.policy))

            linear_system = np.zeros((self.states, self.states))
            res = np.zeros(self.states)

            for state in range(self.states):
                if state == self.goal:
                    linear_system[state][state] = 1
                    continue

                """
                Vs = 1 + Vs' -> -1 = -Vs + Vs'
                """

                res[state] = -self.costs[state][self.policy[state]]
                linear_system[state][state] = -1
                for succ, prob in self.transitions[state][self.policy[state]]:
                    linear_system[state][succ] += prob

            self.V = np.linalg.solve(linear_system, res)
            old_policy = self.policy.copy()
            self.update_policy()
        self.run_time = time.time() - start_time
        return self.policy

    def modified_policy_iteration(self):
        start_time = time.time()
        self._init_var(True)

        # self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]

        # old_policy = [-1 for _ in range(self.states)]
        old_policy = np.full(self.states, -1, dtype=int)

        while not np.array_equal(old_policy, self.policy):
            self.iter += 1
            # print(self.iter, np.sum(old_policy == self.policy))
            # new_V = self.V.copy()

            self.evaluate_policy()

            self.V = self.next_V.copy()
            # self.V, self.aux_V = self.aux_V, self.V
            # print(np.array_equal(self.V, self.aux_V))
            # old_policy = deepcopy(self.policy)
            # old_policy, self.policy = self.policy, old_policy
            old_policy = self.policy.copy()

            self.update_policy()

        self.run_time = time.time() - start_time
        return self.policy

    def update_policy(self):
        for state in range(self.states):
            self.policy[state] = self._bellman_backup(state)[0]

    def evaluate_policy(self):
        for state in range(self.states):
            self.next_V[state] = self.costs[state][self.policy[state]] + sum(self.V[succ] * prob
                                                                             for succ, prob in
                                                                             self.transitions[state][
                                                                                 self.policy[state]])
            # self.next_V[state] = self._bellman_backup(state)[1]

    def get_policy(self):
        return list(self.policy)

    def get_initial_policy(self):
        if self.initial_policy is not None:
            return list(self.initial_policy)
        return self.initial_policy

    def get_V(self):
        return self.V

    def get_stats(self):
        return {'Iterations': self.iter,
                'Run_Time:': self.run_time,
                'Avg_Time': self.run_time / self.iter}


def read_states():
    global states_name, states_idx, mdp_problem, grid_idx
    states_name.clear()
    states_idx.clear()
    line = input_file.readline().strip()
    while not line.startswith('end'):
        states_name.extend([name for name in line.split(', ')])
        line = input_file.readline().strip()

    size = max(get_state_pos(states_name[-1])) + 1
    grid_idx = [[-1 for _ in range(size)] for _ in range(size)]
    for state in range(len(states_name)):
        x, y = get_state_pos(states_name[state])
        grid_idx[x][y] = state

    for i, state in enumerate(states_name):
        states_idx[state] = i

    mdp_problem.set_state_num(len(states_idx))


def read_action(name):
    global mdp_problem
    action_id = actions_name[name]
    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, dest, prob = line.split()[:3]
        mdp_problem.add_action(states_idx[src], states_idx[dest], action_id, float(prob))
        line = input_file.readline().strip()


def read_cost():
    global mdp_problem
    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, action, cost = line.split()
        mdp_problem.add_cost(states_idx[src], actions_name[action], float(cost))
        line = input_file.readline().strip()


def read_state(initial=True):
    state = states_idx[input_file.readline().strip()]
    # print(state)
    if initial:
        mdp_problem.set_initial_state(state)
    else:
        mdp_problem.set_goal(state)
    input_file.readline()


def read_input():
    try:
        while True:
            line = input_file.readline().strip()
            if line.startswith('states'):
                read_states()
            elif line.startswith('action'):
                read_action(line.split()[1])
            elif line.startswith('cost'):
                read_cost()
            elif line.startswith('initial'):
                read_state()
            elif line.startswith('goal'):
                read_state(False)
            elif line.startswith('Grid'):
                break

    except EOFError:
        pass


def get_state_pos(name: str):
    p = name.rsplit('y', 1)
    p[0] = p[0].rsplit('x', 1)[1]

    return int(p[1]) - 1, int(p[0]) - 1


def to_str(old):
    return str(old).replace('.', ',')


def print_res(policy, input_path, iteration):
    print(f'Executada {iteration}')
    output_file.write(f'{iteration}\n')

    # size = max(get_state_pos(states_name[-1])) + 1

    stats = mdp_problem.get_stats()
    stats['S0'] = states_name[mdp_problem.initial_state]
    stats['Goal'] = states_name[mdp_problem.goal]

    for key, val in stats.items():
        output_file.write(f'{key}: {val}\n')

    if mdp_problem.get_initial_policy() is not None:
        output_file.write(f'\nInitial Grid\n')
        policy_to_grid(mdp_problem.get_initial_policy())
        # print_grid(policy_to_grid(mdp_problem.get_initial_policy(), size))

    output_file.write(f'\nFinal Grid:\n')
    policy_to_grid(policy)

    output_file.write(f'\nV:\n')
    output_file.write(' '.join(state.center(18) for state in states_name) + '\n')
    output_file.write(' '.join(['{:.4f}'.format(i).center(18) for i in mdp_problem.get_V()]) + '\n\n')

    stats_file.write('{};{};{};'.format(iteration, input_path, len(grid_idx)) +
                     ';'.join(to_str(val) for val in stats.values()) + '\n')


def policy_to_grid(policy):
    global grid_idx
    size = len(grid_idx)
    grid = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(len(grid_idx[i])):
            if grid_idx[i][j] != -1:
                grid[i][j] = actions[policy[grid_idx[i][j]]]

    # for idx, action in enumerate(policy):
    #     x, y = get_state_pos(states_name[idx])
    #     grid[x][y] = actions[action]
    # print(grid)
    grid = grid[::-1]

    for line in grid:
        output_file.write(' '.join(str(item) for item in line) + '\n')
    # return grid


def solve(input_path):
    read_input()
    print_res(mdp_problem.value_iteration(), input_path, 'Value Iteration')
    print_res(mdp_problem.modified_policy_iteration(), input_path, 'Modified Policy Iteration')
    # print_res(mdp_problem.policy_iteration(), input_path, 'Policy Iteration')


# def get_basedir_name(dir_path):
#     return path.basename(path.dirname(dir_path))
#
#
# # TODO: Fix args receiving
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input_file",
#                 default='C:\\Users\\Gabriel\\Documents\\Grid\\',
#                 help=f"arquivo de entrada para leitura")
# ap.add_argument("-o", "--output_file",
#                 help=f"arquivo de saida para salvar grids")
# ap.add_argument("-st", "--stats_file", default='stats.csv',
#                 help=f"arquivo para armazenar estatisticas que serao usadas para gerar graficos")
# args = vars(ap.parse_args())
# #
# # input_folder = path.realpath(args['input_file'])
# input_folder = args['input_file']
# # if 'output_file' in args:
# #     output_file = args['output_file']
# # else:
# #     output_file = path.join(path.dirname(__file__), path.basename(args['input_file']).rsplit('.', 1)[0] + '.txt')
#
# # output_file = open(output_file, 'w')
# output_file = open(path.join(path.dirname(__file__), 'output.txt'), 'w')
# stats_file = open(path.join(path.dirname(__file__), args['stats_file']), 'w')
#
# solve(path.basename(path.dirname(args['input_file'])), args['input_file'].rsplit('.', 1)[0].rsplit('_', 1)[1])

stats_file = open(path.realpath('stats2.csv'), 'w')
if path.isdir(input_dir):
    files = [path.join(dp, f) for dp, dn, filenames in walk(input_dir)
             for f in filenames if path.splitext(f)[1].lower() == '.net']
else:
    files = [input_dir]

try:
    for input_path in files:
        print(f'Executando MDP para {input_path}')
        mdp_problem = MDP()
        input_file = open(input_path)
        output_file = open(path.splitext(input_path)[0] + '_out.txt', 'w')
        solve(input_path)
        input_file.close()
        output_file.close()

except Exception as err:
    print(err)

stats_file.close()
