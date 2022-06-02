MAP = [
    "XXXXXXXXXXX",
    "XR  X    GX",
    "X   XX    X",
    "X         X",
    "X         X",
    "X     XX  X",
    "X X   X   X",
    "X X     XXX",
    "XYX   XB  X",
    "XXXXXXXXXXX"
]

# Номера действий
ACTIONS = dict(SOUTH=0, NORTH=1, WEST=3, EAST=2,
               PICKUP=4, DROP_OFF=5)

# Возможные награды(за каждый ход, за плохое обращение с клиентом и за победу)
REWARDS = dict(DEFAULT=-1, ERROR=-10, WIN=20)

# Цвета для отрисовки
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

# Гипер-параметры алгоритма
Q_LEARNING_PARAMS = dict(alpha=0.1,
                         gamma=0.6,
                         epsilon=0.1,
                         delta=0.1,
                         max_epochs=1000,
                         max_games=1000000)
