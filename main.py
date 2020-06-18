import argparse
from os import path
from mdp import MDP

states_idx = dict()
states_name = []
actions_name = {'move-south': 0,
                'move-north': 1,
                'move-west': 2,
                'move-east': 3}

actions = {-1: 0,
           0: 'v',
           1: '^',
           2: '<',
           3: '>'}

mdp_problem = MDP()


# input_file = open('G:\\Meu Drive\\Study\\SI\\2020.1\\Planejamento\\EP2\\TestesGrid\\FixedGoalInitialState
# \\navigation_1.net')


def read_states():
    global states_name, states_idx, mdp_problem
    states_name = []
    line = input_file.readline().strip()
    while not line.startswith('end'):
        states_name.extend([name for name in line.split(', ')])
        line = input_file.readline().strip()

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


def print_res(policy, folder_name, file_number, iteration):
    height, width = get_state_pos(states_name[-1])
    grid = [[0 for _ in range(width + 1)] for _ in range(height + 1)]
    for idx, action in enumerate(policy):
        x, y = get_state_pos(states_name[idx])
        grid[x][y] = actions[action]

    # print(grid)
    grid = grid[::-1]

    output_file.write(f'{iteration}\n')

    for key, val in mdp_problem.get_stats().items():
        output_file.write(f'{key}: {val}\n')

    # output_file.write(f'{mdp_problem.get_stats()["Iterations"]} {mdp_problem.get_stats()["Run_Time"]} \n')
    for line in grid:
        output_file.write(' '.join(str(item) for item in line) + '\n')
        # print(' '.join(str(item) for item in line))

    stats = mdp_problem.get_stats()
    stats['Iteration'] = iteration
    stats['Folder'] = folder_name
    stats['File'] = file_number
    # for key, val in stats:
    stats_file.write(';'.join(to_str(val) for val in stats.values()) + '\n')


def solve(folder_name, file_number):
    read_input()
    print_res(mdp_problem.value_iteration(), folder_name, file_number, 'Value Iteration')
    print_res(mdp_problem.policy_iteration(), folder_name, file_number, 'Policy Iteration')


# TODO: Fix args receiving

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_file",
                default='C:\\Users\\Gabriel\\Documents\\Grid\\FixedGoalInitialState\\navigation_1.net',
                help=f"arquivo de entrada para leitura")
ap.add_argument("-o", "--output_file",
                help=f"arquivo de saida para salvar grids")
ap.add_argument("-st", "--stats_file", default='stats.csv',
                help=f"arquivo para armazenar estatisticas que serao usadas para gerar graficos")
# ap.add_argument("-s", "--silent", default=False, action='store_true',
#                 help=f"executar sem impressoes para tela")
args = vars(ap.parse_args())

input_file = open(args['input_file'])
# if 'output_file' in args:
#     output_file = args['output_file']
# else:
#     output_file = path.join(path.dirname(__file__), path.basename(args['input_file']).rsplit('.', 1)[0] + '.txt')

# output_file = open(output_file, 'w')
output_file = open(path.join(path.dirname(__file__), 'output.txt'), 'w')
stats_file = open(path.join(path.dirname(__file__), args['stats_file']), 'w')

solve(path.basename(path.dirname(args['input_file'])), args['input_file'].rsplit('.', 1)[0].rsplit('_', 1)[1])
stats_file.close()
output_file.close()
