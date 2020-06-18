import time
import numpy as np


class MDP:

    def __init__(self, num_states=0, epsilon=0.1):
        self.V = None
        self.epsilon = epsilon
        self.iter = self.initial_state = self.goal = self.run_time = -1

        if num_states != 0:
            self.set_state_num(num_states)

        else:
            self.states = num_states
            self.transitions = self.costs = None

        self.policy = None

    def set_state_num(self, num_states):
        self.states = num_states
        self.transitions = [[[] for _ in range(4)] for _ in range(self.states)]
        self.costs = [[1.0 for _ in range(4)] for _ in range(self.states)]

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
                Q_s[action] = self.costs[state][action]
                for successor, probability in self.transitions[state][action]:
                    Q_s[action] += self.V[successor] * probability

        return int(np.argmin(Q_s)), float(np.min(Q_s))

    def value_iteration(self):
        start_time = time.time()
        self.V = np.ones(self.states)
        self.V[self.goal] = 0
        self.iter = 0
        max_res = self.epsilon

        # TODO: Implement bar logging
        while max_res >= self.epsilon:
            max_res = 0
            new_V = self.V.copy()
            self.iter += 1

            for state in range(self.states):
                new_V[state] = self._bellman_backup(state)[1]
                max_res = max(max_res, abs(new_V[state] - self.V[state]))

            # max_res = np.max(np.abs(new_V-self.V))
            self.V = new_V

        self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]
        self.run_time = time.time() - start_time
        return self.policy

    def _get_proper_action(self, state):
        for idx, action in enumerate(self.transitions[state]):
            if len(action) >= 1:
                for successors in action:
                    if successors[0] != state:
                    # (action[0][0] != state or len(action) > 1):
                        return idx
        return 0

    def policy_iteration(self):
        start_time = time.time()
        self.V = np.ones(self.states)
        self.V[self.goal] = 0
        self.iter = 0

        self.policy = [self._get_proper_action(state) for state in range(self.states)]
        self.policy[self.goal] = -1

        # self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]

        old_policy = [-1 for _ in range(self.states)]

        while old_policy != self.policy:
            self.iter += 1

            new_V = self.V.copy()

            for state in range(self.states):
                new_V[state] = self._bellman_backup(state)[1]

            self.V = new_V
            # old_policy = deepcopy(self.policy)
            old_policy, self.policy = self.policy, old_policy

            for state in range(self.states):
                self.policy[state] = self._bellman_backup(state)[0]

        self.run_time = time.time() - start_time
        return self.policy

    def get_policy(self):
        return self.policy

    def get_stats(self):
        return {'Iterations': self.iter,
                'Run_Time:': self.run_time,
                'Avg_Time': self.run_time / self.iter}

