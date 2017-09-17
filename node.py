import numpy as np
import seaborn as sns
from graphics import Rectangle, Point
import time


def gen_rectangle(o, x, y, l, color='black', dx=100, dy=100):
    p1 = Point(x*dx, y*dy)
    if o == 0:  # horizontal
        p2 = Point(x*dx + dx*l, (1+y)*dy)
    else:
        p2 = Point((1+x)*dx, y*dy + dy*l)

    rectangle = Rectangle(p1, p2)
    rectangle.setFill(color)

    return rectangle


class Node():

    def __init__(self, state):
        self.state = state
        self.g = np.inf
        self.h = np.inf
        self.f = np.inf
        self.parent = None
        self.children = []

        # n_colors = len(self.state)
        # self.color_palette = ['#ff0000'] + sns.color_palette('muted', n_colors).as_hex()

    ###def __repr__(self):
    ###    row_candidates = [len(x) for x in self.state[0]]
    ###    col_candidates = [len(y) for y in self.state[1]]
    ###    return f'node, \nrow_candidates: {row_candidates},\ncol_candidates: {col_candidates}\n'

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)
    #     import pdb; pdb.set_trace()
    #     pass
    #     return self.state == other.state

    def __neq__(self, other):
        return not self.__eq__(other)

    def get_ancestors(self):
        chain = [self]
        current = self
        while current.parent:
            current = current.parent
            chain.append(current)

        # if len(chain) != 1:
        #     chain.append(current)

        return chain

    def visualize_state(self, window):
        while window.items:
            window.items[0].undraw()

        rectangles = []

        for (state, color) in zip(self.state, self.color_palette):
            rectangle = gen_rectangle(*state, color=color)
            rectangles.append(rectangle)
            rectangle.draw(window)

        # time.sleep(0.1)

    def visualize_trace(self, window):
        for each in self.get_ancestors()[::-1]:
            import pdb; pdb.set_trace()
            each.visualize_state(window)
