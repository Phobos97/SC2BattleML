from math import sqrt

from s2clientprotocol.raw_pb2 import DisplayType, Unit
from sc2.ids.unit_typeid import UnitTypeId

from BattleML_definitions.unit_enum import UnitsML, UNIT_TYPE_COUNT, UNITS_ML_DICT, FLAG_BUILDING, \
    PRESERVATION_BUILDING_IDS
from s2clientprotocol import common_pb2
import numpy as np
from typing import List

from bitstring import BitArray
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch

def filter_unit(unit: Unit) -> bool:
    # Alliance 2 is ally, 3 is neutral
    if unit.alliance == 3 or unit.alliance == 2:
        return False

    # the building used to keep the match from ending should be undetectable to the bots.
    if unit.unit_type in PRESERVATION_BUILDING_IDS:
        return False

    try:
        temp = UnitsML(unit.unit_type).name
        # print(temp)
    except ValueError:
        print(f"Encountered unexpected unit: {unit.unit_type} - {UnitTypeId(unit.unit_type).name}")
        return False

    return unit.display_type == DisplayType.Visible


def raw_units_processing(raw_observation, max_units) -> (torch.Tensor, torch.Tensor):
    all_units: List[Unit] = raw_observation.units

    all_units = [unit for unit in all_units if filter_unit(unit)]

    own_units = [unit for unit in all_units if unit.alliance == 1]
    enemy_units = [unit for unit in all_units if unit.alliance == 4]

    own_flag = [unit for unit in own_units if unit.unit_type == FLAG_BUILDING.value]
    enemy_flag = [unit for unit in enemy_units if unit.unit_type == FLAG_BUILDING.value]

    nr_own_units = len(own_units)
    nr_enemy_units = len(enemy_units)

    nr_own_flag = len(own_flag)
    nr_enemy_flag = len(enemy_flag)

    if len(own_flag) > 0:
        own_flag_position = (own_flag[0].pos.x, own_flag[0].pos.y)
    else:
        own_flag_position = (0, 0)

    own_tags = [unit.tag for unit in own_units]
    own_tags = own_tags[0:max_units] if len(own_tags) > max_units else own_tags
    enemy_tags = [unit.tag for unit in enemy_units]
    enemy_tags = enemy_tags[0:max_units] if len(enemy_tags) > max_units else enemy_tags

    processed_own_units = [process_unit(unit, own_flag_position) for unit in own_units]
    processed_own_units = processed_own_units[0:max_units] if len(processed_own_units) > max_units else processed_own_units
    processed_own_units = processed_own_units + [[0]*UNIT_ENCODING_SIZE] * (max_units - len(processed_own_units))

    processed_enemy_units = [process_unit(unit, own_flag_position) for unit in enemy_units]
    processed_enemy_units = processed_enemy_units[0:max_units] if len(processed_enemy_units) > max_units else processed_enemy_units
    processed_enemy_units = processed_enemy_units + [[0] * UNIT_ENCODING_SIZE] * (max_units - len(processed_enemy_units))

    processed_own_units = torch.tensor(processed_own_units)
    processed_enemy_units = torch.tensor(processed_enemy_units)
    scalar_features = torch.tensor([nr_own_units, nr_enemy_units, nr_own_flag, nr_enemy_flag])

    return processed_own_units, processed_enemy_units, scalar_features, (own_tags, enemy_tags)

# nr of different scalar features the above function produces
SCALAR_FEATURE_SIZE = 4

def process_unit(unit: Unit, flag_position) -> List:
    unit_type_one_hot = [0]*UNIT_TYPE_COUNT
    unit_type_one_hot[UNITS_ML_DICT[unit.unit_type]] = 1

    hp_one_hot = [0] * 18
    hp_one_hot[int(sqrt(min(unit.health + unit.shield, 300)))] = 1

    health_ratio = unit.health/unit.health_max
    shield_ratio = 0 if unit.shield_max == 0 else unit.shield / unit.shield_max
    energy_ratio = 0 if unit.energy_max == 0 else unit.energy / unit.energy_max
    weapon_cooldown = unit.weapon_cooldown / 10

    x_position_binary = list(map(int, list(format(int(unit.pos.x), '08b'))))
    y_position_binary = list(map(int, list(format(int(unit.pos.y), '08b'))))

    # TODO cloak, current order, buffs, etc
    # TODO add upgrades?

    # flag is only relevant for certain scenarios
    distance_to_flag = ((flag_position[0] - unit.pos.x)**2 + (flag_position[1] - unit.pos.y)**2)**0.5 / 100

    processed_data = unit_type_one_hot + x_position_binary + y_position_binary + [unit.pos.x/255, unit.pos.y/255] + [health_ratio, shield_ratio, energy_ratio, weapon_cooldown] + hp_one_hot + [distance_to_flag]

    return processed_data


UNIT_ENCODING_KEY = {"unit_type": 0, "x_pos_binary": UNIT_TYPE_COUNT, "y_pos_binary": UNIT_TYPE_COUNT + 8,
                     "x_pos": UNIT_TYPE_COUNT + 16, "y_pos": UNIT_TYPE_COUNT + 17, "health_ratio": UNIT_TYPE_COUNT + 18,
                     "shield_ratio": UNIT_TYPE_COUNT + 19, "energy_ratio": UNIT_TYPE_COUNT + 20,
                     "weapon_cooldown": UNIT_TYPE_COUNT + 21, "hp_one_hot": UNIT_TYPE_COUNT + 22,
                     "distance_to_flag": UNIT_TYPE_COUNT + 40}

UNIT_ENCODING_SIZE = max(UNIT_ENCODING_KEY.values()) + 1

def raw_image_processing(raw_image_observation) -> torch.Tensor:
    # separate own and enemy player units into separate layers
    relative_layer: common_pb2.ImageData = raw_image_observation.player_relative
    relative_layer = np.fromstring(relative_layer.data, dtype=np.uint8).reshape(relative_layer.size.x, relative_layer.size.y)
    own_units_layer = np.zeros_like(relative_layer)
    own_units_layer[relative_layer == 1] = 1
    enemy_units_layer = np.zeros_like(relative_layer)
    enemy_units_layer[relative_layer == 4] = 1

    own_units_layer = torch.tensor(own_units_layer)
    enemy_units_layer = torch.tensor(enemy_units_layer)

    # height map, scaled to range from [1-0]
    height_layer: common_pb2.ImageData = raw_image_observation.height_map
    height_layer = torch.tensor(np.fromstring(height_layer.data, dtype=np.uint8).reshape(height_layer.size.x, height_layer.size.y) / 255)

    # pathable layer, needs to be converted from single bits, pathable/un-pathable with values 1/0
    pathable_layer: common_pb2.ImageData = raw_image_observation.pathable
    pathable_bits = BitArray(bytes=pathable_layer.data)
    pathable_layer = torch.tensor(np.fromstring(pathable_bits.bin, dtype='u1').reshape(pathable_layer.size.x, pathable_layer.size.y))

    # indicates terrain that is in view/explored/unexplored with values 1/0.5/0
    visibility_layer: common_pb2.ImageData = raw_image_observation.visibility_map
    visibility_layer = torch.tensor(np.fromstring(visibility_layer.data, dtype=np.uint8).reshape(visibility_layer.size.x, visibility_layer.size.y) / 2)

    # stack all the layers
    combined_layers = torch.stack([own_units_layer, enemy_units_layer, height_layer, pathable_layer, visibility_layer], dim=0).float()

    return combined_layers

# nr of different spacial features the above function produces
SPACIAL_FEATURES = 5


