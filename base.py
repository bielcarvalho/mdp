import argparse
from os import path, walk, makedirs
from typing import Optional, List, TextIO
import numpy as np
from mdp import MDP

states_idx = dict()
states_name = []
grid_idx = []
actions_idx = {
    'move-south': 0,
    'move-north': 1,
    'move-west': 2,
    'move-east': 3,
    'south': 0,
    'north': 1,
    'west': 2,
    'east': 3,
    '-': -1
}

actions_symbol = {-1: 'G', 0: 'v', 1: '^', 2: '<', 3: '>'}
is_grid_world = mdp_problem = input_file = input_path = output_path = input_dir = stats_file = None


def read_states():
    """ Reads the states from the input file and updates the number on the MDP """
    global states_name, states_idx, mdp_problem, grid_idx
    states_name.clear()
    states_idx.clear()
    line = input_file.readline().strip()
    while not line.startswith('end'):
        states_name.extend([name for name in line.split(', ')])
        line = input_file.readline().strip()

    if is_grid_world:
        size = max(get_state_pos(states_name[-1])) + 1
        grid_idx = [[-1 for _ in range(size)] for _ in range(size)]
        for state in range(len(states_name)):
            x, y = get_state_pos(states_name[state])
            grid_idx[x][y] = state

    for i, state in enumerate(states_name):
        states_idx[state] = i

    mdp_problem.set_state_num(len(states_idx))


def read_action(action_name: str):
    """ Reads the transitions related to action_name from the input file and inserts them on the MDP """
    global mdp_problem
    if is_grid_world:
        action_id = actions_idx[action_name]
    else:
        action_id = mdp_problem.actions
        actions_idx[action_name] = action_id
        actions_symbol[action_id] = action_name
    mdp_problem.add_action()

    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, dest, prob = line.split()[:3]
        mdp_problem.add_transition(states_idx[src], states_idx[dest],
                                   action_id, float(prob))
        line = input_file.readline().strip()


def read_cost():
    """ Reads the transition costs from the input file and inserts them on the MDP """
    global mdp_problem
    mdp_problem.new_cost()
    line = input_file.readline().strip()
    while not line.startswith('end'):
        src, action, cost = line.split()
        mdp_problem.add_cost(states_idx[src], actions_idx[action], float(cost))
        line = input_file.readline().strip()


# def read_state(initial: bool = True):
#     state = states_idx[input_file.readline().strip()]
#     if initial:
#         mdp_problem.set_initial_state(state)
#     else:
#         mdp_problem.set_goal(state)
#     input_file.readline()


def read_input():
    """ Reads the data from the input file and sets up the MDP """
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
                state = states_idx[input_file.readline().strip()]
                mdp_problem.set_initial_state(state)
                # read_state()
            elif line.startswith('goal'):
                state = states_idx[input_file.readline().strip()]
                mdp_problem.set_goal(state)
                # read_state(False)
            elif line.startswith('Grid') or line.startswith('Policy'):
                break

    except EOFError:
        pass
    actions_idx['-'] = -1


def get_state_pos(name: str):
    p = name.rsplit('y', 1)
    p[0] = p[0].rsplit('x', 1)[1]

    return int(p[1]) - 1, int(p[0]) - 1


def to_str(old):
    return str(old).replace('.', ',')


def print_res(policy: list, iteration_type: str):
    """ Prints the output data to the output files """
    print(f'Executada {iteration_type}')
    output_file = open(output_path + f'_{iteration_type}.txt', 'w')

    stats = mdp_problem.get_stats()
    stats['S0'] = states_name[mdp_problem.initial_state]
    stats['Goal'] = states_name[mdp_problem.goal]

    for key, val in stats.items():
        output_file.write(f'{key}: {val}\n')

    if mdp_problem.get_initial_policy() is not None:
        output_file.write(f'\nInitial Grid:\n' if is_grid_world else f'\nInitial Policy:\n')
        print_policy(mdp_problem.get_initial_policy(), output_file)

    output_file.write(f'\nFinal Grid:\n' if is_grid_world else f'\nFinal Policy:\n')
    print_policy(policy, output_file, mdp_problem.get_V())

    file_name = (path.relpath(input_path, input_dir)
                 if path.relpath(input_path, input_dir) != '.'
                 else path.basename(input_path))

    stats_file.write('{};{};'.format(iteration_type, file_name) +
                     ';'.join(to_str(val) for val in stats.values()) + '\n')

    output_file.close()


def print_policy(policy: list,
                 output_file: TextIO,
                 v: Optional[np.ndarray] = None):
    """ Prints the policy and V to the output file """
    if is_grid_world:
        grid_print(policy, output_file, v)
    else:
        max_name_len = max(6, len(states_name[-1]))
        output_file.write(' '.join(state.center(max_name_len) for state in states_name) + '\n')
        output_file.write(' '.join(actions_symbol[i].center(max_name_len) for i in policy) + '\n')
        if v is not None:
            output_file.write(' '.join(['{:.2f}'.format(i).center(max_name_len) for i in v]) + '\n')
        output_file.write('\n')


def grid_print(policy: list,
               output_file: TextIO,
               v: Optional[np.ndarray] = None):
    """ Prints the policy and V in a grid format (for grid world problem) to the output file """
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
    output_file.write(' '.join('x{}'.format(item).center(max_name_len)
                               for item in range(1, size + 1)) + '\n')

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
        output_file.write(' '.join('x{}'.format(item).center(max_name_len)
                                   for item in range(1, size + 1)) + '\n')
    output_file.write('\n')


def equiv_policy(p1: list, p2: list, v: Optional[np.ndarray]) -> bool:
    """ Checks if policies' actions are equivalent according to V """
    def get_succ_val(pol, st_idx):
        a = pol[st_idx]
        succ = list(mdp_problem.probability_transitions[a][st_idx].nonzero()[1])
        if not succ:
            return -1
        return mdp_problem.costs[a][st_idx] + sum(
            mdp_problem.probability_transitions[a][st_idx, s] * v[s]
            for s in succ)

    equiv = True

    for i in range(len(p1)):
        if p1[i] != p2[i]:
            v1, v2 = get_succ_val(p1, i), get_succ_val(p2, i)
            if v1 != v2:
                equiv = False
                print(f'{i} ({states_name[i]}): {actions_symbol[p1[i]]} - {v1}, {actions_symbol[p2[i]]} - {v2}')

    return equiv


def read_initial_policy() -> Optional[List[int]]:
    """ Reads a initial proper policy from json file, used by the policy iteration algorithms """
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
    """ Gets the last folder name from path_dir """
    read_input()

    initial_policy = read_initial_policy()
    if vi:
        p1 = mdp_problem.value_iteration()
        print_res(p1, 'VI')

    # v = mdp_problem.get_V().copy()

    if pi:
        p2 = mdp_problem.policy_iteration(initial_policy)
        print_res(p2, 'PI')

    if mpi:
        p3 = mdp_problem.modified_policy_iteration(initial_policy)
        print_res(p3, 'MPI')

    # print('PI == VI' if p1 == p2 else f'PI ~ VI: {equiv_policy(p1, p2, v)}')
    # print('PI == MPI' if p2 == p3 else f'PI ~ MPI: {equiv_policy(p2, p3, v)}')


def get_basedir_name(dir_path: str) -> str:
    """ Gets the last folder name from path_dir """
    return path.basename(path.dirname(dir_path))


def main():
    """ Receives input arguments, and sets up files and MDP """
    global is_grid_world, mdp_problem, input_file, input_path, output_path, input_dir, stats_file
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",
                    "--input_path",
                    default='TestesGrid',
                    help=f"arquivo ou pasta de entrada para leitura")
    ap.add_argument("-o",
                    "--output_path",
                    default=None,
                    help=f"pasta para salvar arquivos de saida")
    ap.add_argument("-gw",
                    "--grid_world",
                    default=False,
                    action='store_true',
                    help=f"usar para representar saida por meio de um grid")
    ap.add_argument("-vi",
                    "--value_iteration",
                    default=False,
                    action='store_true',
                    help=f"executar iteracao de valor")
    ap.add_argument("-pi",
                    "--policy_iteration",
                    default=False,
                    action='store_true',
                    help=f"executar iteracao de politica")
    ap.add_argument("-mpi",
                    "--modified_policy_iteration",
                    default=False,
                    action='store_true',
                    help=f"executar iteracao de politica modificada")
    ap.add_argument("-e",
                    "--epsilon",
                    default=0.1,
                    type=float,
                    help=f"epsilon desejado (padrao 0,1)")
    args = vars(ap.parse_args())

    assert args['value_iteration'] or args['policy_iteration'] or args['modified_policy_iteration'], \
        'Ao menos um algoritmo de iteracao deve ser selecionado'

    is_grid_world = args['grid_world']
    input_dir = path.realpath(args['input_path'])

    if args['output_path'] is None:
        output_dir = None
        stats_path = path.join((input_dir
                                if path.isdir(input_dir)
                                else path.dirname(input_dir)),
                               'stats.csv')
    else:
        output_dir = path.realpath(args['output_path'])
        makedirs(output_dir, exist_ok=True)
        stats_path = path.join(output_dir, 'stats.csv')

    if path.exists(stats_path):
        stats_file = open(stats_path, 'a')
    else:
        stats_file = open(stats_path, 'w')
        stats_file.write(
            'Iteration_Type;Input_File;States;Iterations;Run_Time(ms);Avg_Time(ms);'
            'Bellman_Time(ms);Solve_Time(ms);Oper_Matrix_Time(ms);'
            'Total_Iterations;Initial_State;Final_State\n')

    if path.isdir(input_dir):
        files = [
            path.join(dp, f)
            for dp, dn, filenames in walk(input_dir)
            for f in filenames
            if path.splitext(f)[1].lower() == '.net'
        ]
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
                output_path = path.join(output_dir,
                                        path.splitext(path.relpath(input_path, input_dir))[0])
                makedirs(path.dirname(output_path), exist_ok=True)

            solve(args['value_iteration'], args['policy_iteration'],
                  args['modified_policy_iteration'])
            input_file.close()
            stats_file.flush()

    except Exception as err:
        print(err)

    stats_file.close()


if __name__ == '__main__':
    main()
