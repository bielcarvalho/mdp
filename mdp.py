import time
from copy import deepcopy

import numpy as np


class MDP:

    def __init__(self, num_states=0, epsilon=0.1):
        self.V = None
        self.epsilon = epsilon
        self.iter = self.initial_state = self.goal = self.run_time = -1

        # self.successors = [[list for _ in range(4)] for _ in range(self.states)]
        # self.probabilities = [[list for _ in range(4)] for _ in range(self.states)]

        # self.successors = np.array(np.full((self.states, 4), np.nan), dtype=int)
        # self.probabilities = np.array(np.full((self.states, 4), np.nan), dtype=float)
        if num_states != 0:
            self.set_state_num(num_states)
            # self.transitions = [[[] for _ in range(4)] for _ in range(self.states)]
            # self.costs = [[1.0 for _ in range(4)] for _ in range(self.states)]

        else:
            self.states = num_states
            self.transitions = self.costs = None

        self.policy = None

    def set_state_num(self, num_states):
        self.states = num_states
        self.transitions = [[[] for _ in range(4)] for _ in range(self.states)]
        self.costs = [[1.0 for _ in range(4)] for _ in range(self.states)]

    def add_action(self, source: int, destination: int, action_idx: int, probability: float):
        # self.transitions[source][action_idx] = (destination, probability, cost)
        # self.transitions[source][action_idx][0] = destination
        # self.transitions[source][action_idx][1] = probability
        self.transitions[source][action_idx].append((destination, probability))
        # self.probabilities[source][action_idx] = probability

    def add_cost(self, source: int, action_idx: int, cost: float):
        self.costs[source][action_idx] = cost

    def get_actions(self, state: int):
        # return [action for action in self.transitions[state] if action is not None]
        pass

    def set_goal(self, new_goal: int):
        self.goal: int = new_goal

    def set_initial_state(self, new_initial_state: int):
        self.initial_state = new_initial_state

    def _bellman_backup(self, state: int):
        if state == self.goal:
            return -1, 0
        Q_s = np.full(4, np.nan)
        # for i in range(4):
        # np.where(self.successors != np.nan, 1+(self.V[self.successors]))
        # Q = np.array(self.successors)
        for action in range(4):
            if len(self.transitions[state][action]) > 0:
                Q_s[action] = self.costs[state][action]
                for successor, probability in self.transitions[state][action]:
                    Q_s[action] += self.V[successor] * probability

        return int(np.argmin(Q_s)), float(np.min(Q_s))

    def value_iteration(self):
        # self.V = [[1 for _ in range(self.width)] for _ in range(self.height)]
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
            # print('Iteracao ', self.iter)
            # print('Residual ', max_res)

            for state in range(self.states):
                new_V[state] = self._bellman_backup(state)[1]
                # print(new_V[state])
                max_res = max(max_res, abs(new_V[state] - self.V[state]))
            # new_V[self.goal] = 0

            # max_res = np.max(np.abs(new_V-self.V))
            self.V = new_V
        print(self.V)
        self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]
        self.run_time = time.time() - start_time

    def _proper_actions(self, state):
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

        # self.policy = [next((idx for idx, action in enumerate(self.transitions[state]) if action and
        #                      action[successors][0] != state for successors in action), 0)
        #                for state in range(self.states)]

        self.policy = [self._proper_actions(state) for state in range(self.states)]
        self.policy[self.goal] = -1
        # for state in range(self.states):
        #     self.V[state] = self._bellman_backup(state)[1]

        # self.policy = [self._bellman_backup(state)[0] for state in range(self.states)]

        old_policy = [-1 for _ in range(self.states)]

        while old_policy != self.policy:
            self.iter += 1
            print('Iteracao {}'.format(self.iter))

            new_V = self.V.copy()

            for state in range(self.states):
                new_V[state] = self._bellman_backup(state)[1]

            # if new_V == self.V:
            #     print(new_V)
            #     exit(1)

            self.V = new_V
            # old_policy = deepcopy(self.policy)
            old_policy, self.policy = self.policy, old_policy

            for state in range(self.states):
                self.policy[state] = self._bellman_backup(state)[0]

        self.run_time = time.time() - start_time

    def get_policy(self):
        return self.policy

    def get_stats(self):
        return {'Iterations': self.iter,
                'Run Time:': self.run_time,
                'Avg Time': self.run_time / self.iter}

