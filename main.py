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
f = open('G:\\Meu Drive\\Study\\SI\\2020.1\\Planejamento\\EP2\\TestesGrid\\FixedGoalInitialState\\navigation_1.net')


def read_states():
    global states_name, states_idx, mdp_problem
    states_name = []
    line = f.readline().strip()
    while not line.startswith('end'):
        states_name.extend([name for name in line.split(', ')])
        line = f.readline().strip()

    # print(states_name)
    for i, state in enumerate(states_name):
        states_idx[state] = i

    mdp_problem.set_state_num(len(states_idx))


# def read_line():
#     line = f.readline().strip()
#     if not line.startswith('end'):
#         return line
#     return ''

def read_action(name):
    global mdp_problem
    action_id = actions_name[name]
    line = f.readline().strip()
    while not line.startswith('end'):
        src, dest, prob = line.split()[:3]
        mdp_problem.add_action(states_idx[src], states_idx[dest], action_id, float(prob))
        line = f.readline().strip()


def read_cost():
    global mdp_problem
    line = f.readline().strip()
    while not line.startswith('end'):
        src, action, cost = line.split()
        mdp_problem.add_cost(states_idx[src], actions_name[action], float(cost))
        line = f.readline().strip()


def read_state(initial=True):
    state = states_idx[f.readline().strip()]
    # print(state)
    if initial:
        mdp_problem.set_initial_state(state)
    else:
        mdp_problem.set_goal(state)
    f.readline()


def read_input():
    try:
        while True:
            line = f.readline().strip()
            print(line)
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


def grid_res(policy):
    height, width = get_state_pos(states_name[-1])
    grid = [[0 for _ in range(width + 1)] for _ in range(height + 1)]
    for idx, action in enumerate(policy):
        x, y = get_state_pos(states_name[idx])
        grid[x][y] = actions[action]

    # print(grid)
    return grid[::-1]


def print_grid(grid):
    for line in (grid):
        # print(line)
        print(' '.join(str(item) for item in line))


read_input()

# mdp_problem.value_iteration()
# grid = grid_res(mdp_problem.get_policy())
# print_grid(grid)

mdp_problem.policy_iteration()
grid = grid_res(mdp_problem.get_policy())
print_grid(grid)

# TODO: Implement file receiving through args with argparse
# TODO: Implement storage for output in files
