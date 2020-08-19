import time
from typing import Optional, List, Tuple, Iterable, Union

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


class MDP:
    """ Solves a stochastic Markov Decision Process with Value and Policy Iteration algorithms """

    def __init__(self,
                 num_states: int = 0,
                 actions: int = 0,
                 transitions: Optional[List[sp.dok_matrix]] = None,
                 predecessors: Optional[List[List[Tuple[int, int]]]] = None,
                 costs: Optional[np.ndarray] = None,
                 epsilon: int = 0.1):
        """ Initializes MDP class.

        With the exception of epsilon, the arguments may be provided here,
        or with the appropriate class' methods.
        Take S as the number of states, and A as the number of actions.
        
        Args:
            num_states:
                Optional; Number of states of the MDP.
            actions:
                Optional; Number of actions of the MDP.
            transitions:
                Optional; List of size A, containing sparse matrices with dimensions (S, S).
                transitions[a][s, s'] represents the probability (float) that an action a,
                applied at a state s, results in state s'.
            predecessors:
                Optional; List of size S containing, for each state, a list of its predecessor
                states and actions. That is, if transitions[a][s, s'] > 0, (s, a) should be in
                predecessors[s']. Should be provided if transitions is not None, to use
                Policy Iteration without providing a proper initial policy.
            costs:
                Optional; Array of dimensions (A, S), representing, for any costs[a, s],
                the cost of applying the cost of action a in state s. Since it's a
                stochastic MDP, when a can't be applied at s, costs[a, s] = np.inf.
            epsilon:
                Optional; Minimum residual (maximum value change between iterations)
                to detect convergence and stop the execution. Used in value iteration
                and modified policy iteration. Default is 0.1.
        """
        self.prev_V = self.V = None
        self.costs = costs
        self.epsilon = epsilon
        self.iter = self.sub_iter = self.initial_state = self.goal = self.run_time = -1
        self.oper_matrix_time = self.bellman_time = self.solve_time = 0
        self.actions = actions
        self.states = num_states
        self.predecessors = predecessors
        self.probability_transitions = [] if transitions is None else transitions

        if num_states != 0:
            self.set_state_num(num_states)

        self.initial_policy = self.policy = self.old_policy = self.policy_cost = self.policy_probability = None

    def set_state_num(self, num_states: int):
        """ Sets the new number of states, and the list of predecessor states accordingly """
        self.states = num_states
        if self.predecessors is None:
            self.predecessors = [[] for _ in range(self.states)]

    def add_action(self):
        """ Increases the number of actions """
        self.actions += 1
        self.probability_transitions.append(sp.dok_matrix((self.states, self.states)))

    def add_transition(self, state_source: int, state_dest: int,
                       action_idx: int, probability: float):
        """
        Adds a new transition probability, and updates the list of predecessors to reflect it.

        Args:
            action_idx:
                Index of the action responsible for the transition.
            state_source:
                State in which the action was applied.
            state_dest:
                State in which the action resulted.
            probability:
                Probability that the action leads to state_dest.
        """
        self.probability_transitions[action_idx][state_source, state_dest] = probability
        self.predecessors[state_dest].append((state_source, action_idx))

    def new_cost(self):
        """ Resets the costs array to the default infinite values """
        self.costs = np.full((self.actions, self.states), np.inf)

    def add_cost(self, state_source: int, action_idx: int, cost: float):
        """
        Sets the cost to apply a specific action.

        Args:
            action_idx:
                Index of the action responsible for the transition.
            state_source:
                State in which the action was applied.
            cost:
                Cost to apply the action on the given state.
        """
        self.costs[action_idx][state_source] = cost

    def set_goal(self, new_goal: int):
        """ Sets the state goal """
        self.goal: int = new_goal
        # To not mess with the value between the iterations, set the cost from actions leaving the goal to 0.
        ind = np.where(self.costs[:, self.goal] == np.inf)
        if ind[0].size > 0:
            self.costs[ind[0], self.goal] = 0

    def set_initial_state(self, new_initial_state: int):
        self.initial_state = new_initial_state

    # def _dummy_bellman_backup(self, state: int):
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

    def _bellman_backup(self, value_only: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
       Calculates Bellman Backup for every action and state.

        Args:
            value_only:
                Optional; Boolean that indicates whether to return just the new V,
                or the new policy as well.

        Returns:
            New V and new policy, or just new V, according to value_only.
        """

        Q = np.empty((self.actions, self.states))
        for action in range(self.actions):
            Q[action] = self.costs[action] + self.probability_transitions[action].dot(self.prev_V)
        if value_only:
            return Q.min(axis=0)
        return Q.min(axis=0), Q.argmin(axis=0)

    def _update_policy_matrices(self, changed_states: Iterable):
        """ Updates policy specific matrices.

        Updates the policy_probability and policy_cost matrices to reflect the actions changed
        in the current policy, used in Policy Iteration and Modified Policy Iteration.
        """
        for state in changed_states:
            # Since updating a sparse matrix would be costly, it just stores in an array the
            # transitions of the changed states using the action in the current policy.
            self.policy_probability[state] = self.probability_transitions[self.policy[state]][state]
            self.policy_cost[state] = self.costs[self.policy[state]][state]

    def _calculate_operation_matrices(self) -> sp.csr_matrix:
        """ Makes probability matrix for vectorized solutions in Policy Iteration.

        Updates the policy_probability and policy_cost arrays to reflect the actions
        from the current policy, and then make a sparse probability matrix for vectorized
        solutions in Policy Iteration and Modified Policy Iteration.

        Returns:
            Sparse probability matrix.
        """
        t1 = time.time()

        self._update_policy_matrices(np.where(self.old_policy != self.policy)[0])

        if self.old_policy[self.goal] != self.policy[self.goal]:
            self.policy_probability[self.goal] = sp.csr_matrix(np.zeros(self.states))
            self.policy_cost[self.goal] = 0

        self.oper_matrix_time += time.time() - t1

        # Concatenates transitions in policy_probability to generate a sparse matrix of size (S, S).
        return sp.bmat(self.policy_probability, format='csr')

    def _init_iter_var(self, initial_policy: Optional[np.ndarray] = None):
        """ Initializes variables used by the solvers.

        Args:
            initial_policy:
                Optional; Initial policy used by the policy iteration algorithms.
        """
        self.V = np.ones(self.states)
        self.V[self.goal] = 0
        self.iter = self.sub_iter = 0
        self.old_policy = np.full(self.states, -1)

        if initial_policy is not None:
            self.policy_probability = np.empty((self.states, 1), dtype=object)
            self.policy_cost = np.zeros(self.states)

            self.initial_policy = np.array(initial_policy)
            self.policy = self.initial_policy.copy()

        if not sp.isspmatrix_csr(self.probability_transitions[0]):
            self.probability_transitions = [t.tocsr() for t in self.probability_transitions]

        self.oper_matrix_time = self.bellman_time = self.solve_time = 0
        self.run_time = time.time()

        if initial_policy is not None:
            self._update_policy_matrices(np.arange(self.states))
            self.policy_probability[self.goal] = sp.csr_matrix(np.zeros(self.states))
            self.policy_cost[self.goal] = 0

    def value_iteration(self) -> List[int]:
        """
        Solves the MDP using a Value Algorithm.

        Returns:
            Final policy.
        """

        self._init_iter_var()
        max_res = self.epsilon

        while max_res >= self.epsilon:
            self.iter += 1
            self.prev_V = self.V.copy()

            self.V = self._bellman_backup(value_only=True)
            max_res = np.max(np.abs(self.V - self.prev_V))

        self.policy = self._bellman_backup()[1]
        return self._end_iter()

    def _end_iter(self):
        self.run_time = time.time() - self.run_time
        self.policy[self.goal] = -1

        return self.get_policy()

    def _make_proper_policy(self) -> np.ndarray:
        """
        Creates a proper initial policy, by iterating through the list of predecessor states.

        Returns:
            Proper initial policy.

        Raises:
            AssertionError:
                If no initial policy is provided, and no list of predecessor states
                was found to create one.
        """
        assert self.predecessors, 'Forneca uma politica inicial ou uma lista de estados antecessores para cria-la'
        policy = np.full(self.states, -1, dtype=int)
        policy[self.goal] = 0

        define_predecessors = [self.goal]
        while define_predecessors:
            state = define_predecessors.pop()
            for predecessor, action in self.predecessors[state]:
                if policy[predecessor] == -1:
                    # policy[predecessor] = self._get_proper_action(predecessor, state)
                    policy[predecessor] = action
                    define_predecessors.append(predecessor)

        policy[self.goal] = -1
        return policy

    # def _get_proper_action(self, state: int, successor: int):
    #     for action in range(self.actions):
    #         if self.probability_transitions[action][state, successor] > 0:
    #             return action
    #     return 0

    def policy_iteration(self, initial_policy: Optional[np.ndarray] = None) -> List[int]:
        """
        Solves the MDP using a Policy Iteration Algorithm.

        Args:
            initial_policy:
                Optional; Array with the size of the number of states, representing the action
                initially taken for each state. It needs to be proper, such that there are no
                cycles to avoid reaching the goal.

        Returns:
            Final policy.

        Raises:
            AssertionError:
                If no initial policy is provided, and no list of predecessor states
                was found to create one.
        """

        if initial_policy is None:
            initial_policy = self._make_proper_policy()
        self._init_iter_var(initial_policy)

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
            self.solve_time += time.time() - t

            self._update_policy()

        return self._end_iter()

    def modified_policy_iteration(self, initial_policy: Optional[np.ndarray] = None) -> List[int]:
        """
        Solves the MDP using a Modified Policy Iteration Algorithm.

        Args:
            initial_policy:
                Optional; Array with the size of the number of states, representing the action
                initially taken for each state. It needs to be proper, such that there are no
                cycles to avoid reaching the goal.

        Returns:
            Final policy.

        Raises:
            AssertionError:
                If no initial policy is provided, and no list of predecessor states
                was found to create one.
        """
        if initial_policy is None:
            initial_policy = self._make_proper_policy()
        self._init_iter_var(initial_policy)

        while not np.array_equal(self.old_policy, self.policy):
            self.iter += 1

            self.iterative_policy_evaluation()

            self._update_policy()

            if np.max(np.abs(self.V - self.prev_V)) < self.epsilon:
                break

        return self._end_iter()

    def _print_iter(self):
        max_name_len = 6  # max(6, len(states_name[-1]))
        print('V{}'.format(self.iter).center(max_name_len) + ' ')
        print(' '.join(['{:.2f}'.format(i).center(max_name_len) for i in self.V]) + '\n')

        print('P{}'.format(self.iter).center(max_name_len) + ' ')
        # print(' '.join(['{}'.format(actions_symbol[i]).center(max_name_len) for i in self.policy]) + '\n')
        print(' '.join(['{}'.format(i).center(max_name_len) for i in self.policy]) + '\n')

    def _update_policy(self):
        self.prev_V = self.V.copy()
        self.old_policy = self.policy.copy()
        t = time.time()
        self.V, self.policy = self._bellman_backup()
        self.bellman_time += time.time() - t
        self.sub_iter += 1

    def iterative_policy_evaluation(self):
        """ Approximates the V calculation iteratively """

        max_res = self.epsilon
        probability = self._calculate_operation_matrices()
        t = time.time()
        while max_res >= self.epsilon:
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
        return {
            'States': self.states,
            'Iterations': self.iter,
            'Run_Time(ms):': self.run_time * 1000,
            'Avg_Time(ms)': self.run_time * 1000 / self.iter,
            'Bellman_Time(ms)': self.bellman_time * 1000 / self.iter,
            'Solve_Time(ms)': self.solve_time * 1000 / self.iter,
            'Oper_Matrix_Time(ms)': self.oper_matrix_time * 1000 / self.iter,
            'Total_Iter': self.sub_iter if self.sub_iter > 0 else self.iter
        }

