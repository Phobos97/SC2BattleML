from typing import List, Union

import numpy as np
from sc2.unit import Unit
from torch import Tensor

from BattleML_definitions.observations import UNIT_ENCODING_KEY
from BattleML_definitions.unit_enum import UNIT_TYPE_COUNT, UNITS_ML_DICT, FLAG_BUILDING
from bots.micro_bot import MicroBot
from scenarios.scenario import Scenario


def filter_units(unit_list: np.array):
    units = []
    for unit in unit_list:
        if not any(unit[:UNIT_TYPE_COUNT]):
            # ignore filler unit
            continue
        if unit[UNITS_ML_DICT[FLAG_BUILDING.value]]:
            # ignore flag
            continue
        units.append(unit)
    return units


def health_ratio_change(prev_units: List[Tensor], cur_units: List[Unit]):
    previous_total_health = 0
    for unit in prev_units:
        previous_total_health += np.array(unit[UNIT_ENCODING_KEY["health_ratio"]].to('cpu'))

    current_total_health = 0
    for unit in cur_units:
        current_total_health += np.array(unit.health/unit.health_max)

    return current_total_health - previous_total_health

def unit_count_change(prev_units: List[Tensor], cur_units: List[Unit]):
    previous_unit_count = len(prev_units)
    current_unit_count = len(cur_units)

    return current_unit_count - previous_unit_count

def check_for_tie(run_manager, scenario: Scenario):
    return run_manager.episode_step_nr > scenario.settings.env_steps_per_episode

def capture_flag_combat_reward(run_manager, bot: MicroBot, scenario) -> (float, float, Union[MicroBot, None]):
    win_reward = 0
    total_reward = 0
    winner = None

    if run_manager.prev_output[bot.name] is None:
        return total_reward, win_reward, winner

    # set enemy bot variable
    enemy_bot = scenario.defender_bot if bot == scenario.attacker_bot else scenario.attacker_bot

    # retrieve unit info from previous frame
    prev_own_units = filter_units(run_manager.prev_output[bot.name][1][0])
    prev_enemy_units = filter_units(run_manager.prev_output[bot.name][1][1])

    # unit health reward
    own_health_change = health_ratio_change(prev_own_units, bot.get_units())
    own_health_reward = own_health_change
    own_health_reward = own_health_reward if abs(own_health_reward) > 0.01 else 0

    enemy_health_change = health_ratio_change(prev_enemy_units, enemy_bot.get_units())
    enemy_health_reward = enemy_health_change * -2
    enemy_health_reward = enemy_health_reward if abs(enemy_health_reward) > 0.01 else 0


    # flag hp reward
    flag_health_reward = 0
    win_flag_reward = 0

    flags = enemy_bot.get_flag()
    for flag in flags:
        flag_health_reward -= flag.health_percentage*0.2

    if len(flags) == 0:
        print("extra flag reward! all enemy flags destroyed!")
        win_flag_reward = 10
        winner = bot


    win_reward += win_flag_reward
    total_reward += own_health_reward + enemy_health_reward + flag_health_reward + distance_reward

    return total_reward, win_reward, winner

def pure_combat_reward(run_manager, bot: MicroBot, scenario) -> (float, float, Union[MicroBot, None]):
    win_reward = 0
    total_reward = 0
    winner = None

    if run_manager.prev_output[bot.name] is None:
        return total_reward, win_reward, winner

    # set enemy bot variable
    enemy_bot = scenario.defender_bot if bot == scenario.attacker_bot else scenario.attacker_bot

    # retrieve unit info from previous frame
    prev_own_units = filter_units(run_manager.prev_output[bot.name][1][0])
    prev_enemy_units = filter_units(run_manager.prev_output[bot.name][1][1])

    # unit count reward
    own_unit_change = unit_count_change(prev_own_units, bot.get_units())
    enemy_unit_change = unit_count_change(prev_enemy_units, enemy_bot.get_units())

    own_unit_reward = 2 * own_unit_change
    enemy_unit_reward = -2 * min(0, enemy_unit_change)

    enemy_health_change = health_ratio_change(prev_enemy_units, enemy_bot.get_units())
    enemy_health_reward = enemy_health_change * -1
    enemy_health_reward = enemy_health_reward if abs(enemy_health_reward) > 0.01 else 0


    enemy_total_health_reward = 0
    for unit in enemy_bot.get_units():
        enemy_total_health_reward -= unit.health_percentage*0.2

    if len(enemy_bot.get_units()) == 0:
        win_reward += 2
        winner = bot
    if len(bot.get_units()) == 0:
        win_reward -= 0.5
        winner = enemy_bot


    total_reward += own_unit_reward + enemy_unit_reward + enemy_health_reward + enemy_total_health_reward

    return total_reward, win_reward, winner
