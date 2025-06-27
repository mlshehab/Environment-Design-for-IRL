from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import random

from env_design.env_params import EnvParams, EnvDesign


@dataclass
class Params(EnvParams):
    disabled: bool = False

    wall_1: bool = False
    wall_1_x1: float = 0
    wall_1_y1: float = 0
    wall_1_x2: float = 0
    wall_1_y2: float = 0

    wall_2: bool = False
    wall_2_x1: float = 0
    wall_2_y1: float = 0
    wall_2_x2: float = 0
    wall_2_y2: float = 0

    wall_3: bool = False
    wall_3_x1: float = 0
    wall_3_y1: float = 0
    wall_3_x2: float = 0
    wall_3_y2: float = 0

    wall_4: bool = False
    wall_4_x1: float = 0
    wall_4_y1: float = 0
    wall_4_x2: float = 0
    wall_4_y2: float = 0


@dataclass
class ParamsDefault(EnvParams):
    disabled: bool = False

    wall_1: bool = False
    wall_1_x1: float = 0
    wall_1_y1: float = 0
    wall_1_x2: float = 0
    wall_1_y2: float = 0

    wall_2: bool = False
    wall_2_x1: float = 0
    wall_2_y1: float = 0
    wall_2_x2: float = 0
    wall_2_y2: float = 0

    wall_3: bool = False
    wall_3_x1: float = 0
    wall_3_y1: float = 0
    wall_3_x2: float = 0
    wall_3_y2: float = 0

    wall_4: bool = False
    wall_4_x1: float = 0
    wall_4_y1: float = 0
    wall_4_x2: float = 0
    wall_4_y2: float = 0

class Bounds(NamedTuple):
    wall_1: tuple = random.choice, [False, True]
    wall_1_x1: tuple = np.random.uniform, -0.1, 0.6
    wall_1_y1: tuple = np.random.uniform, -0.1, 0.6
    wall_1_x2: tuple = np.random.uniform, -0.1, 0.6
    wall_1_y2: tuple = np.random.uniform, -0.1, 0.6

    wall_2: tuple = random.choice, [False, True]
    wall_2_x1: tuple = np.random.uniform, -0.1, 0.6
    wall_2_y1: tuple = np.random.uniform, -0.1, 0.6
    wall_2_x2: tuple = np.random.uniform, -0.1, 0.6
    wall_2_y2: tuple = np.random.uniform, -0.1, 0.6

    wall_3: tuple = random.choice, [False, True]
    wall_3_x1: tuple = np.random.uniform, -0.1, 0.6
    wall_3_y1: tuple = np.random.uniform, -0.1, 0.6
    wall_3_x2: tuple = np.random.uniform, -0.1, 0.6
    wall_3_y2: tuple = np.random.uniform, -0.1, 0.6

    wall_4: tuple = random.choice, [False, True]
    wall_4_x1: tuple = np.random.uniform, -0.1, 0.6
    wall_4_y1: tuple = np.random.uniform, -0.1, 0.6
    wall_4_x2: tuple = np.random.uniform, -0.1, 0.6
    wall_4_y2: tuple = np.random.uniform, -0.1, 0.6



class MazeDesign(EnvDesign):
    env = "MazeED"
    params = Params
    default = ParamsDefault
    bounds = Bounds
    path: str = "maze/"

    EXPERT_EPISODE_RETURN = 1000.
