import time
from abc import ABC, abstractmethod

from node import Node


class A_Star(ABC):

    def __init__(self, heuristic_fun, window=None, as_dfs=False):
        """
         heuristic_fun: Node -> float, a function that estimates
         the remaining length until reaching goal node.

         as_dfs: Boolean, if True use a LIFO-queue instead of a sorted queue.
        """
        self.solution = None
        self.known_nodes = []
        self.closed_queue = []
        self.open_queue = []
        self.window = window
        self.as_dfs = as_dfs

        self.heuristic_fun = heuristic_fun

    @abstractmethod
    def cost_fun(self, parent, child):
        """returns the exact cost of going from `parent` to `child`"""
        pass

    @abstractmethod
    def goal_fun(self, node):
        """
        Returns True if `node` has reached the goal. Otherwise returns False
        """
        pass

    @abstractmethod
    def generate_children(self, parent):
        """
        parent: a Node-instance

        generates children for each parent by applying all possible operators on
        parent.

        Returns an iterator that yields new children for parent. Each child
        should be called self._attach_and_eval(child, parent) in this function
        to properly set `parent` as its parent.
        """
        pass

    def _attach_and_eval(self, child, parent):
        """
        attach `child` to its `parent` and evaluates the f, g, h values of
        child in the following manner:

            child.h = self.heuristic_fun(child)
            child.g = parent.g + self.cost_fun(parent, child)
            child.f = child.g + child.h

        returns:
         child - the modified Node instance
        """

        child.parent = parent
        child.g = parent.g + self.cost_fun(parent, child)
        child.h = self.heuristic_fun(child)

        child.f = child.g + child.h

        return child

    def _propagate_path_improvements(self, parent):
        """
        from parent, propagate new path improvements (if any)
        on its children. Also causes changes, so the heuristic function
        h is needed.

        input:
         parent: a node instance
         heuristic_fun: a function heuristic_fun(node) -> float

        returns:
         None

        """
        for child in parent.children:
            candidate_g = parent.g + self.cost_fun(parent, child)
            if candidate_g < child.g:
                child.parent = parent

                child.g = candidate_g
                child.f = child.g + self.heuristic_fun(child)

    def _make_initial_node(self, initial_state):
        """
        For initial state, creates a Node instance with that state with
        node.g = 0, node.h = self.heuristic_fun(node).

        returns the Node-instance
        """
        node = Node(initial_state)
        node.info = 0
        node.g = 0
        node.h = self.heuristic_fun(node)
        node.f = node.g + node.h
        return node

    def _step(self, node):
        """
        Executes a single step of the a* algorithm.

        node: the node popped from `self.open_queue` to be processed.

        returns None, but might alter nodes in the graph.
        """
        def state_exists_and_assign(node):
            """
            Sets `node` as the previous existing node if in self.known_nodes
            and returns True, otherwise returns False.
            """
            if any([node == each for each in self.known_nodes]):
                idx = [each.state for each in self.known_nodes].index(node.state)
                visited_node = self.known_nodes[idx]
                node = visited_node

                import pdb; pdb.set_trace()
                return True
            else:
                return False

        for each in self.generate_children(node):
            has_changed = state_exists_and_assign(each)

            if not has_changed:  # that means it is new
                self.known_nodes.append(each)

            if each not in self.closed_queue and each not in self.open_queue:
                self._attach_and_eval(each, node)
                if self.as_dfs:
                    self.open_queue = [each] + self.open_queue  # prepend
                else:
                    self.open_queue.append(each)

            if node.g + self.cost_fun(node, each) < each.g:
                self._attach_and_eval(each, node)

                if each in self.closed_queue:
                    self._propagate_path_improvements(each)

    def solve(self, initial_state, visual=False):
        initial_node = self._make_initial_node(initial_state)

        self.open_queue.append(initial_node)
        self.known_nodes = [initial_node]

        i = 0
        while self.open_queue:
            node = self.open_queue.pop(0)

            if self.window:
                node.visualize_state(self.window)

            self.closed_queue.append(node)

            if not self.as_dfs:  # sort if it's not dfs
                self.open_queue.sort(key=lambda N: N.f, reverse=False)

            print(f'step{i}\t\tqueue size {len(self.open_queue)}\t\th={node.h}')
            i += 1

            if self.goal_fun(node):
                self.solution = node

                return node

            self._step(node)

        raise Exception('No solution is found')

    def print_summary(self):
        n_moves = len(self.solution.get_ancestors()) - 1  # initial is not a move
        n_expand = len(self.closed_queue)
        n_created = n_expand + len(self.open_queue)

        print('\n\n---------------------')
        print('        summary')
        print('---------------------')
        print(f'path length: {n_moves}\nnodes created: {n_created}\nnodes visited: {n_expand}')
        return None
