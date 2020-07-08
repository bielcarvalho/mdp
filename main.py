import argparse
import time
from copy import deepcopy
from os import path, walk, makedirs
import numpy as np

from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

states_idx = dict()
states_name = []
grid_idx = []
actions_idx = {'move-south': 0,
               'move-north': 1,
               'move-west': 2,
               'move-east': 3,
               'south': 0,
               'north': 1,
               'west': 2,
               'east': 3,
               '-': -1}

actions_symbol = {-1: 'G',
                  0: 'v',
                  1: '^',
                  2: '<',
                  3: '>'
                  }

input_dir = path.realpath('s68.net')


class MDP:

    def __init__(self, num_states=0, actions=0, transitions=None, predecessors=None, epsilon=0.1):
        self.prev_V = self.V = self.costs = None
        self.epsilon = epsilon
        self.iter = self.sub_iter = self.initial_state = self.goal = self.run_time = -1
        self.oper_matrix_time = self.bellman_time = self.solve_time = 0
        self.actions = actions
        self.states = num_states
        self.predecessors = predecessors
        self.transitions = [] if transitions is None else transitions
        # self.transitions_bkp = None

        if predecessors is None and num_states != 0:
            self.predecessors = [[] for _ in range(self.states)]


        self.initial_policy = self.policy = self.old_policy = self.policy_cost = self.policy_probability = None

    def set_state_num(self, num_states):
        self.states = num_states
        self.predecessors = [[] for _ in range(self.states)]
        # self.transitions_bkp = [[] for _ in range(self.states)]

        # if GRID_WORLD:
        #     self.transitions = [[[] for _ in range(4)] for _ in range(self.states)]
        #     self.predecessors = [[] for _ in range(self.states)]
        # else:
        #     self.transitions, self.predecessors = [[[] for _ in range(self.states)] for _ in range(2)]
        #     self.costs = None

    def add_action_layer(self):
        self.actions += 1
        # for s in range(self.states):
        #     self.transitions_bkp[s].append([])
        self.transitions.append(sp.dok_matrix((self.states, self.states)))

    def add_action(self, source: int, destination: int, action_idx: int, probability: float):
        self.transitions[action_idx][source, destination] = probability
        # self.transitions_bkp[source][action_idx].append((destination, probability))
        self.predecessors[destination].append(source)

    def new_cost(self):
        # self.costs = np.zeros((self.states, self.actions))
        # self.costs = np.zeros((self.actions, self.states))
        self.costs = np.full((self.actions, self.states), np.inf)
        self.costs[:, self.goal] = 0


    def add_cost(self, source: int, action_idx: int, cost: float):
        # self.costs[source][action_idx] = cost
        self.costs[action_idx][source] = cost

    def set_goal(self, new_goal: int):
        self.goal: int = new_goal

    def set_initial_state(self, new_initial_state: int):
        self.initial_state = new_initial_state

    # def _bellman_backup(self, state: int):
    #     if state == self.goal:
    #         return -1, 0
    #
    #     Q_s = np.full(self.actions, np.inf)
    #
    #     for action in range(self.actions):
    #         if self.transitions[state][action]:
    #             Q_s[action] = self.costs[state][action] + sum(self.prev_V[succ] * prob
    #                                                           for succ, prob in self.transitions[state][action])
    #     return int(np.argmin(Q_s)), float(np.min(Q_s))

    def _bellman_backup(self, value_only=False):
        Q = np.empty((self.actions, self.states))
        for action in range(self.actions):
            Q[action] = self.costs[action] + self.transitions[action].dot(self.prev_V)
        if value_only:
            return np.min(Q, axis=0)
        return np.min(Q, axis=0), np.argmin(Q, axis=0)

    def _calculate_operation_matrices(self):
        t1 = time.time()

        changes = np.where(self.old_policy != self.policy, self.policy, -1)
        for action in range(self.actions):
            src_states_idx = (changes == action).nonzero()[0]

            if src_states_idx.size > 0:
                self.policy_probability[src_states_idx] = self.transitions[action][src_states_idx]
                # self.policy_probability[src_states_idx] = self.transitions[action][src_states_idx].todense()
                self.policy_cost[src_states_idx] = self.costs[action][src_states_idx]

        if changes[self.goal] != -1:
            self.policy_probability[self.goal] = 0
            self.policy_cost[self.goal] = 0

        self.oper_matrix_time += time.time() - t1

        return self.policy_probability.tocsr()

    def _init_var(self):
        self.V = np.ones(self.states)
        # self.V = np.zeros(self.states)
        self.V[self.goal] = 0
        self.iter = self.sub_iter = 0
        self.old_policy = np.full(self.states, -1)
        self.policy_probability = sp.lil_matrix((self.states, self.states))
        # self.policy_probability = np.zeros((self.states, self.states))
        self.policy_cost = np.zeros(self.states)

        if not sp.isspmatrix_csr(self.transitions[0]):
            self.transitions = [sp.csr_matrix(t) for t in self.transitions]

        # max_name_len = max(6, len(states_name[-1]))
        # print(' '.center(max_name_len) + ' ' + ' '.join(state.center(max_name_len) for state in states_name))

        self.oper_matrix_time = self.bellman_time = self.solve_time = 0
        self.run_time = time.time()

    def value_iteration(self):
        self._init_var()
        max_res = self.epsilon

        # TODO: Implement bar logging
        while max_res >= self.epsilon:
            self.iter += 1
            self.prev_V = self.V.copy()

            # max_res = 0
            #
            # for state in range(self.states):
            #     self.V[state] = self._bellman_backup(state)[1]
            #     max_res = max(max_res, abs(self.V[state] - self.prev_V[state]))

            self.V = self._bellman_backup(value_only=True)
            max_res = np.max(np.abs(self.V - self.prev_V))
            # self.print_iter()

        # self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]
        self.policy = self._bellman_backup()[1]
        return self.end_iter()

    def end_iter(self):
        self.run_time = time.time() - self.run_time
        self.policy[self.goal] = -1

        return self.get_policy()

    def _make_proper_policy(self):
        policy = np.full(self.states, -1, dtype=int)
        policy[self.goal] = 0

        define_predecessors = [self.goal]
        while define_predecessors:
            state = define_predecessors.pop()
            for predecessor in self.predecessors[state]:
                if policy[predecessor] == -1:
                    policy[predecessor] = self._get_proper_action(predecessor, state)
                    define_predecessors.append(predecessor)

        policy[self.goal] = -1
        self.policy = policy
        self.initial_policy = self.policy.copy()
        return policy

    def _get_proper_action(self, state, successor):
        for action in range(self.actions):
            if self.transitions[action][state, successor] > 0:
                return action
        # for idx, action in enumerate(self.transitions[state]):
        #     if len(action) >= 1:
        #         for transition in action:
        #             if transition[0] == successor:
        #                 return idx
        return 0

    def policy_iteration(self, initial_policy=None):
        self._init_var()

        if initial_policy is not None:
            self.initial_policy = np.array(initial_policy)
            self.policy = self.initial_policy.copy()
        else:
            self._make_proper_policy()

        while not np.array_equal(self.old_policy, self.policy):
            self.iter += 1

            # linear_system = np.zeros((self.states, self.states))
            # res = np.zeros(self.states)
            #
            # for state in range(self.states):
            #     if state == self.goal:
            #         linear_system[state][state] = 1
            #         continue
            #
            #     res[state] = -self.costs[state][self.policy[state]]
            #     linear_system[state][state] = -1
            #     for succ, prob in self.transitions[state][self.policy[state]]:
            #         linear_system[state][succ] += prob
            # self.V = np.linalg.solve(linear_system, res)



            probability = self._calculate_operation_matrices()
            t = time.time()
            self.V = spsolve(sp.eye(self.states, self.states) - probability, self.policy_cost)
            # self.V = np.linalg.solve(sp.eye(self.states, self.states) - self.policy_probability, self.policy_cost)
            self.solve_time += time.time() - t

            self.update_policy()


        return self.end_iter()

    def modified_policy_iteration(self, initial_policy=None):
        self._init_var()

        if initial_policy is not None:
            self.initial_policy = np.array(initial_policy)
            self.policy = self.initial_policy.copy()
        else:
            self._make_proper_policy()

        while not np.array_equal(self.old_policy, self.policy):  # DO-WHILE
            self.iter += 1

            self.iterative_policy_evaluation()

            self.update_policy()

            if np.max(np.abs(self.V - self.prev_V)) < self.epsilon:
                break

        return self.end_iter()

    def print_iter(self):
        max_name_len = max(6, len(states_name[-1]))
        print('V{}'.format(self.iter).center(max_name_len) + ' ')
        print(' '.join(['{:.2f}'.format(i).center(max_name_len) for i in self.V]) + '\n')

        print('P{}'.format(self.iter).center(max_name_len) + ' ')
        print(' '.join(['{}'.format(actions_symbol[i]).center(max_name_len) for i in self.policy]) + '\n')

    def update_policy(self):
        self.prev_V = self.V.copy()
        self.old_policy = self.policy.copy()
        t = time.time()
        self.V, self.policy = self._bellman_backup()
        self.bellman_time += time.time() - t
        self.sub_iter += 1

        # for state in range(self.states):
        #     Q_s = self._bellman_aux(state)
        #     new_action = int(np.argmin(Q_s))
        #     self.next_V[state] = float(Q_s[new_action])
        #     if Q_s[new_action] < Q_s[self.policy[state]]:
        #         self.policy[state] = new_action
        #
        #     # self.policy[state], self.next_V[state] = self._bellman_backup(state)

    def iterative_policy_evaluation(self):
        max_res = self.epsilon
        probability = self._calculate_operation_matrices()
        t = time.time()
        while max_res >= self.epsilon:
            # max_res = 0
            #
            # for state in range(self.states):
            #     if state != self.goal:
            #         self.next_V[state] = self.costs[state][self.policy[state]] + sum(self.V[succ] * prob
            #                                                                          for succ, prob in
            #                                                                          self.transitions[state][
            #                                                                              self.policy[state]])
            #     else:
            #         self.next_V[state] = 0
            #     max_res = max(max_res, abs(self.next_V[state] - self.V[state]))

            self.prev_V = self.V.copy()
            self.V = self.policy_cost + probability.dot(self.prev_V)
            max_res = np.max(np.abs(self.V - self.prev_V))

            self.sub_iter += 1
        self.solve_time += time.time() - t

    def get_policy(self):
        return list(self.policy)

    def get_initial_policy(self):
        if self.initial_policy is not None:
            return list(self.initial_policy)
        return self.initial_policy

    def get_V(self):
        return self.V

    def get_stats(self):
        return {'States': self.states,
                'Iterations': self.iter,
                'Run_Time(ms):': self.run_time * 1000,
                'Avg_Time(ms)': self.run_time * 1000 / self.iter,
                'Bellman_Time(ms)': self.bellman_time * 1000 / self.iter,
                'Solve_Time(ms)': self.solve_time * 1000 / self.iter,
                'Oper_Matrix_Time(ms)': self.oper_matrix_time * 1000 / self.iter,
                'Total_Iter': self.sub_iter if self.sub_iter > 0 else self.iter
                }


def read_states():
    global states_name, states_idx, mdp_problem, grid_idx
    states_name.clear()
    states_idx.clear()
    line = input_file.readline().strip()
    while not line.startswith('end'):
        states_name.extend([name for name in line.split(', ')])
        line = input_file.readline().strip()

    if GRID_WORLD:
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
    if GRID_WORLD:
        action_id = actions_idx[name]
    else:
        action_id = len(actions_idx)
        actions_idx[name] = action_id
        actions_symbol[action_id] = name
    mdp_problem.add_action_layer()

    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, dest, prob = line.split()[:3]
        mdp_problem.add_action(states_idx[src], states_idx[dest], action_id, float(prob))
        line = input_file.readline().strip()


def read_cost():
    global mdp_problem
    mdp_problem.new_cost()
    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, action, cost = line.split()
        mdp_problem.add_cost(states_idx[src], actions_idx[action], float(cost))
        line = input_file.readline().strip()


def read_state(initial=True):
    state = states_idx[input_file.readline().strip()]
    if initial:
        mdp_problem.set_initial_state(state)
    else:
        mdp_problem.set_goal(state)
    input_file.readline()


def read_input():
    try:
        if not GRID_WORLD:
            actions_idx.clear()
            # actions = {-1: 'G'}
            actions_symbol.clear()
            actions_symbol[-1] = 'G'

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
            elif line.startswith('Grid') or line.startswith('Policy'):
                break

    except EOFError:
        pass


def get_state_pos(name: str):
    p = name.rsplit('y', 1)
    p[0] = p[0].rsplit('x', 1)[1]

    return int(p[1]) - 1, int(p[0]) - 1


def to_str(old):
    return str(old).replace('.', ',')


def print_res(policy, iteration_type):
    print(f'Executada {iteration_type}')
    output_file = open(output_path + f'_{iteration_type}.txt', 'w')

    stats = mdp_problem.get_stats()
    stats['S0'] = states_name[mdp_problem.initial_state]
    stats['Goal'] = states_name[mdp_problem.goal]

    for key, val in stats.items():
        output_file.write(f'{key}: {val}\n')

    if mdp_problem.get_initial_policy() is not None:
        output_file.write(f'\nInitial Grid:\n' if GRID_WORLD else f'\nInitial Policy:\n')
        print_policy(mdp_problem.get_initial_policy(), output_file)

    output_file.write(f'\nFinal Grid:\n' if GRID_WORLD else f'\nFinal Policy:\n')
    print_policy(policy, output_file, mdp_problem.get_V())

    file_name = (path.relpath(input_path, input_dir)
                 if path.relpath(input_path, input_dir) != '.'
                 else path.basename(input_path))

    stats_file.write('{};{};'.format(iteration_type, file_name) +
                     ';'.join(to_str(val) for val in stats.values()) + '\n')

    output_file.close()


def print_policy(policy, output_file, v=None):
    if GRID_WORLD:
        grid_print(policy, output_file, v)
    else:
        max_name_len = max(6, len(states_name[-1]))
        output_file.write(' '.join(state.center(max_name_len) for state in states_name) + '\n')
        output_file.write(' '.join(actions_symbol[i].center(max_name_len) for i in policy) + '\n')
        if v is not None:
            output_file.write(' '.join(['{:.2f}'.format(i).center(max_name_len) for i in v]) + '\n')
        output_file.write('\n')


def grid_print(policy, output_file, v=None):
    global grid_idx
    size = len(grid_idx)
    policy_grid = [['0' for _ in range(size)] for _ in range(size)]

    max_name_len = 6
    if v is not None:
        v_grid = [[-1 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(len(grid_idx[i])):
            if grid_idx[i][j] != -1:
                policy_grid[i][j] = actions_symbol[policy[grid_idx[i][j]]]
                if v is not None:
                    v_grid[i][j] = v[grid_idx[i][j]]

    policy_grid = policy_grid[::-1]
    output_file.write(' '.center(max_name_len) + ' ')
    output_file.write(' '.join('x{}'.format(item).center(max_name_len) for item in range(1, size + 1)) + '\n')

    idx = [i for i in range(size, 0, -1)]
    for i, line in enumerate(policy_grid):
        output_file.write('y{}'.format(idx[i]).center(max_name_len) + ' ')
        output_file.write(' '.join(item.center(max_name_len) for item in line) + '\n')

    if v is not None:
        v_grid = v_grid[::-1]
        for i, line in enumerate(v_grid):
            output_file.write('y{}'.format(idx[i]).center(max_name_len) + ' ')
            output_file.write(' '.join('{:.2f}'.format(item).center(max_name_len) for item in line) + '\n')
        output_file.write(' '.center(max_name_len) + ' ')
        output_file.write(' '.join('x{}'.format(item).center(max_name_len) for item in range(1, size + 1)) + '\n')
    output_file.write('\n')


def equiv_policy(p1, p2, v):
    def get_succ_val(pol, st_idx):
        a = pol[st_idx]
        succ = list(mdp_problem.transitions[a][st_idx].nonzero()[1])
        if not succ:
            return -1
        return mdp_problem.costs[a][st_idx] + sum(mdp_problem.transitions[a][st_idx, s] * v[s] for s in succ)

    equiv = True

    for i in range(len(p1)):
        if p1[i] != p2[i]:
            # print(f'{i} ({states_name[i]}): {actions_symbol[p1[i]]}, {actions_symbol[p2[i]]}')
            v1, v2 = get_succ_val(p1, i), get_succ_val(p2, i)
            if v1 != v2:
                equiv = False
                print(f'{i} ({states_name[i]}): {actions_symbol[p1[i]]} - {v1}, {actions_symbol[p2[i]]} - {v2}')

    return equiv


def read_initial_policy():
    import json
    json_path = input_path + '_politicas.json'
    if not path.exists(json_path):
        print('Json nao encontrado, criando politica inicial propria')
        return None

    initial_policy = [-1 for _ in range(mdp_problem.states)]

    with open(json_path) as json_file:
        data = json.load(json_file)

        for state_name, action in data.items():
            initial_policy[states_idx[state_name]] = actions_idx[action]

    return initial_policy


def solve(vi=True, pi=True, mpi=False):
    read_input()
    # if not GRID_WORLD:
    #     print(actions_symbol)

    initial_policy = read_initial_policy()

    if vi:
        p1 = mdp_problem.value_iteration()
        # print_res(mdp_problem.value_iteration(), 'VI')
        print_res(p1, 'VI')

    v = mdp_problem.get_V().copy()

    if pi:
        p3 = mdp_problem.policy_iteration(initial_policy)
        # print_res(mdp_problem.policy_iteration(initial_policy), 'PI')
        print_res(p3, 'PI')

    #
    if mpi:
        p2 = mdp_problem.modified_policy_iteration(initial_policy)
        # print_res(mdp_problem.modified_policy_iteration(initial_policy), 'MPI')
        print_res(p2, 'MPI')


    print('PI == VI' if p1 == p3 else f'PI ~ VI: {equiv_policy(p1, p3, v)}')
    print('PI == MPI' if p2 == p3 else f'PI ~ MPI: {equiv_policy(p2, p3, v)}')

    # if p1 == p2:
    #     print('Equal:', p1 == p2)
    # else:
    #     print('Equiv:', equiv_policy(p1, p2, v))


def get_basedir_name(dir_path):
    return path.basename(path.dirname(dir_path))


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", default='TestesGrid',
                help=f"arquivo ou pasta de entrada para leitura")
ap.add_argument("-o", "--output_path", default=None,
                help=f"pasta para salvar arquivos de saida")
ap.add_argument("-g", "--generic_mdp", default=False, action='store_true',
                help=f"usar para problemas que os estados nao estao dispostos em um grid")
ap.add_argument("-vi", "--value_iteration", default=False, action='store_true',
                help=f"executar iteracao de valor")
ap.add_argument("-pi", "--policy_iteration", default=False, action='store_true',
                help=f"executar iteracao de politica")
ap.add_argument("-mpi", "--modified_policy_iteration", default=False, action='store_true',
                help=f"executar iteracao de politica modificada")
ap.add_argument("-e", "--epsilon", default=0.1, type=float,
                help=f"taxa de erro maxima aceitavel")
args = vars(ap.parse_args())

assert args['value_iteration'] or args['policy_iteration'] or args['modified_policy_iteration'], 'Ao menos um algoritmo de iteracao deve ser selecionado'

GRID_WORLD = not args['generic_mdp']
input_dir = path.realpath(args['input_path'])

if args['output_path'] is None:
    output_dir = None
    stats_path = path.join((input_dir if path.isdir(input_dir) else path.dirname(input_dir)),
                           'stats.csv')
else:
    output_dir = path.realpath(args['output_path'])
    makedirs(output_dir, exist_ok=True)
    stats_path = path.join(output_dir, 'stats.csv')

if path.exists(stats_path):
    stats_file = open(stats_path, 'a')
else:
    stats_file = open(stats_path, 'w')
    stats_file.write('Iteration_Type;Input_File;States;Iterations;Run_Time(ms);Avg_Time(ms);'
                     'Bellman_Time(ms);Solve_Time(ms);Oper_Matrix_Time(ms);'
                     'Total_Iterations;Initial_State;Final_State\n')

if path.isdir(input_dir):
    files = [path.join(dp, f) for dp, dn, filenames in walk(input_dir)
             for f in filenames if path.splitext(f)[1].lower() == '.net']
else:
    files = [input_dir]

try:
    for input_path in files:
        print(f'Executando MDP para {input_path}')
        mdp_problem = MDP(epsilon=args['epsilon'])
        input_file = open(input_path)

        if output_dir is None:
            output_path = path.splitext(input_path)[0]
        else:
            output_path = path.join(output_dir, path.splitext(path.relpath(input_path, input_dir))[0])
            makedirs(path.dirname(output_path), exist_ok=True)
            # output_file = path.join(output_path, path.basename(path.dirname(input_path)) + '_' +
            #                         path.basename(path.splitext(input_path)[0]) + '_out.txt')
        # print('Saida: ', output_file)
        # output_file = open(output_file, 'w')
        solve(args['value_iteration'], args['policy_iteration'], args['modified_policy_iteration'])
        input_file.close()

except Exception as err:
    print(err)

stats_file.close()
