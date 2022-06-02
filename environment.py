import numpy as np
import six
from six import StringIO
from contextlib import closing
from constants import MAP
from constants import ACTIONS as ACT
from constants import REWARDS as R
from constants import color2num


class TaxiEnv:
    def __init__(self):
        # Параметры MDP:
        self.s = None  # текущее состояние
        self.actions_num = len(ACT)
        self.desc = np.asarray(MAP, dtype='c')
        self.rows_num, self.cols_num, self.locations, self.states_num\
            = self.map_reader()
        # Переменные для уменьшения объёма кода
        max_row = self.rows_num - 1
        max_col = self.cols_num - 1
        # Матрица переходов, в терминах MDP это две функции: P и R
        self.P = {state: {action: []
                          for action in range(self.actions_num)}
                  for state in range(self.states_num)}
        # Список возможных начальных состояний
        # (не все состояния могут быть начальными)
        self.initial_states_list = []
        # Параметры для отрисовки
        self.last_action = None
        # Заполнение матрицы переходов
        for state in range(self.states_num):
            destination_idx, pass_loc, col, row = self.decode(state)

            if pass_loc < len(self.locations) \
                    and pass_loc != destination_idx \
                    and self.desc[row + 1, col + 1] != b"X":
                self.initial_states_list.append(state)

            for action in range(self.actions_num):
                new_row, new_col, new_pass_loc = \
                    row, col, pass_loc
                reward = R['DEFAULT']
                done = False
                taxi_loc = (row, col)

                if action == ACT['SOUTH'] \
                        and self.desc[row + 2, col + 1] != b"X":
                    new_row = min(row + 1, max_row)
                elif action == ACT['NORTH'] \
                        and self.desc[row, col + 1] != b"X":
                    new_row = max(row - 1, 0)
                elif action == ACT['EAST'] \
                        and self.desc[1 + row, col + 2] != b"X":
                    new_col = min(col + 1, max_col)
                elif action == ACT['WEST'] \
                        and self.desc[1 + row, col] != b"X":
                    new_col = max(col - 1, 0)
                if action == ACT['PICKUP']:
                    if pass_loc < len(self.locations) \
                            and taxi_loc == self.locations[pass_loc]:
                        new_pass_loc = len(self.locations)
                    else:
                        reward = R['ERROR']
                elif action == ACT['DROP_OFF']:
                    if taxi_loc == self.locations[destination_idx] \
                            and pass_loc == len(self.locations):
                        new_pass_loc = destination_idx
                        done = True
                        reward = R['WIN']
                    elif taxi_loc in self.locations \
                            and pass_loc == len(self.locations):
                        new_pass_loc = self.locations.index(taxi_loc)
                    else:
                        reward = R['ERROR']
                new_state = self.encode(
                    new_row, new_col, new_pass_loc, destination_idx)
                self.P[state][action].append(
                    (new_state, reward, done))

    def reset(self):
        self.s = np.random.choice(self.initial_states_list)
        return self.s

    def step(self, action):
        transition = self.P[self.s][action]
        s, r, d = transition[0]
        self.last_action = action
        self.s = s
        return s, r, d

    def map_reader(self):
        out = [len(self.desc) - 2, len(self.desc[0]) - 2]  # считаем количество строк и столбцов
        locations = []
        for i in range(out[0] + 2):
            for j in range(out[1] + 2):
                point = self.desc[i, j]
                if point != b"X" and point != b" ":
                    locations.append((i - 1, j - 1))  # записываем координаты остановок
        out.append(locations)
        out.append(out[0]*out[1]*len(out[2])*(len(out[2]) + 1))  # количество состояний
        return out

    def encode(self, taxi_row, taxi_col, pass_loc, destination_idx):
        i = taxi_row
        i *= self.cols_num
        i += taxi_col
        i *= len(self.locations) + 1
        i += pass_loc
        i *= len(self.locations)
        i += destination_idx
        return i

    def decode(self, i):
        out = [i % len(self.locations)]
        i = i // len(self.locations)
        out.append(i % (len(self.locations) + 1))
        i = i // (len(self.locations) + 1)
        out.append(i % self.cols_num)
        i = i // self.cols_num
        out.append(i)
        return out

    def render(self):
        outfile = StringIO()

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        destination_idx, pass_idx, taxi_col, taxi_row = self.decode(self.s)

        def ul(x): return "." if x == " " else x
        if pass_idx < 4:
            out[1 + taxi_row][taxi_col + 1] = colorize(
                out[1 + taxi_row][taxi_col + 1], 'yellow', highlight=True)
            pi, pj = self.locations[pass_idx]
            out[1 + pi][pj + 1] = colorize(out[1 + pi][pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][taxi_col + 1] = colorize(
                ul(out[1 + taxi_row][taxi_col + 1]), 'green', highlight=True)

        di, dj = self.locations[destination_idx]
        out[1 + di][dj + 1] = colorize(out[1 + di][dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Drop_off"][self.last_action]))
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(six.u(str(num)))
    if bold:
        attr.append(six.u('1'))
    attrs = six.u(';').join(attr)
    return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)
