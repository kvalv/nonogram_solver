import numpy as np
import drawing
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

    def __eq__(self, other):
        return np.array_equal(self.state, other.state)

    def __neq__(self, other):
        return not self.__eq__(other)

    def get_ancestors(self):
        chain = [self]
        current = self
        while current.parent:
            current = current.parent
            chain.append(current)
        return chain

    def visualize_state(self, window):
        drawing.draw_state(self.state, window)
        time.sleep(0.1)
        # window.getMouse()
        # window.close()
