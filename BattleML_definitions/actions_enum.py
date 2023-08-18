import enum


class ActionsML(enum.Enum):
    # basic commands
    HOLD_POSITION = 0
    MOVE = 1
    ATTACK_TARGET = 2
    ATTACK_MOVE = 3
    NO_OPERATION = 4


ACTION_COUNT = len(ActionsML.__members__)


