import numpy as np
from graphics import Point, GraphWin
from graphics import *

def make_rectangle(x, y, color='black', len_x=1, len_y=1, dx=10, dy=10):
    p1 = Point(x*dx, y*dy)
    p2 = Point((x+len_x)*dx, (y+len_y)*dy)

    r =  Rectangle(p1, p2)
    r.setFill(color)
    return r

def generate_window(x_dim, y_dim):
    x_dim = x_dim * 10
    y_dim = y_dim * 10

    return GraphWin('my window', x_dim, y_dim)

def draw_line(window, line, index, horizontal=True, color='black', opacity=100):
    for idx, each in enumerate(line):
        if each == 1:
            if horizontal==False:
                rectangle = make_rectangle(idx, index, color)
                rectangle.draw(window)
            else:
                rectangle = make_rectangle(idx, index, color)
                rectangle.draw(window)
    pass

def draw_state(state, window):
    while window.items:
        window.items[0].undraw()

    rows, cols = state
    def filter_line(lines):
        for i, each in enumerate(lines):
            if len(each) > 1:
                # import pdb; pdb.set_trace()
                lines[i] = [np.zeros(len(each[0]))]
        return lines

    rows, cols = filter_line(rows), filter_line(cols)

    rows = np.rot90([np.fliplr(e) for e in rows], k=2)
    # rows = [np.fliplr(e) for e in rows]
    cols = np.rot90(np.matrix(np.array([e for e in cols]),),k=3)
    C = cols
    L = []
    for each in C:
        L.append(np.tile(np.asarray(each)[0], [1,1]))
    L = np.array(L)
    
    def draw_lines_in_direction(lines, color='black', horizontal=True):
        if horizontal==False:
            pass
            # lines = np.rot90(np.matrix(lines), k=3)
        for idx, candidates in enumerate(lines):
            if len(candidates) == 1:
                the_line = candidates[0]
                draw_line(window, the_line, idx, horizontal=horizontal,color=color)

            else:
                for each in candidates:
                    draw_line(window, each, idx, color=color, opacity=10, horizontal=horizontal)

    draw_lines_in_direction(rows, color='blue', horizontal=True)
    draw_lines_in_direction(L, color='red', horizontal=False)
    pass
    


def visualize_state(self, window):
    while window.items:
        window.items[0].undraw()
                                                               
    rectangles = []
                                                               
    for (state, color) in zip(self.state, self.color_palette):
        rectangle = gen_rectangle(*state, color=color)
        rectangles.append(rectangle)
        rectangle.draw(window)
