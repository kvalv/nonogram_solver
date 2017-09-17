import numpy as np

import inspect
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class FrameWrapper():
    def __init__(self, frame):
        self.frame = frame
        code = self.frame.f_code
        code_info = str(self.frame.f_code)
        self.line_no = code_info.split(' ')[-1]
        self.func_name = code_info.split(' ')[2]
        self.filename = code_info.split(' ')[6]

        for (k,v) in self.frame.f_locals.items():
            setattr(self, k, v)

        self.args_repr = color.GREEN + '\n'.join([f'{str(k)}: {str(v)}' for (k,v) in self.frame.f_locals.items()]) + color.END

    def __repr__(self):
        return f'{color.CYAN + self.func_name + color.END} at {self.filename} line {self.line_no}\n{self.args_repr}\n'

    def parent(self):
        all_frames = inspect.getouterframes(self.frame)
        if len(all_frames) > 1:
            return FrameWrapper(all_frames[1].frame)


def FUN(combined, func_name):
    """func_name is parent_func name
    probably won't work if recursive.."""
    frame = inspect.currentframe()
    # frame = FUNC(verbose=False)
    X = combined
    # X = [each for each in inspect.getouterframes(frame)]
    idxes = np.where(np.array([each.function == func_name for each in X]))[0]
    if len(idxes) >= 0:
        frame= X[idxes[-1]].frame
        return FrameWrapper(frame)

    return None
