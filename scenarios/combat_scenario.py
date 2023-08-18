import random

from sc2.ids.unit_typeid import UnitTypeId

from SC2BattleML.BattleML_definitions.unit_enum import FLAG_BUILDING
from SC2BattleML.bots.micro_bot import MicroBot
from SC2BattleML.scenarios.combat_reward import capture_flag_combat_reward, pure_combat_reward
from SC2BattleML.scenarios.scenario import Scenario


class CaptureFlagCombatScenario(Scenario):
    """
    Combat Scenario where destroying the enemy flag structure is the winning objective.
    """

    def __init__(self, scenario_settings, attacker_bot: MicroBot, defender_bot: MicroBot, attacker_units: dict = None, defender_units: dict = None):
        super().__init__(scenario_settings=scenario_settings)
        self.name = f"CaptureFlagCombatScenario"
        self.done = False
        self.attacker_bot = attacker_bot
        self.defender_bot = defender_bot

        if attacker_units is None:
            attacker_units = {UnitTypeId.MARAUDER: 2}
        if defender_units is None:
            defender_units = {UnitTypeId.MARAUDER: 2}

        self.attacker_units = attacker_units
        self.defender_units = defender_units

        self.reward_function = capture_flag_combat_reward

    async def get_reward(self, run_manager, bot: MicroBot) -> float:
        reward, win_reward, winner = self.reward_function(run_manager, bot, self)
        if win_reward > 0:
            self.done = True
            self.winner = winner.name
        return reward + win_reward

    async def reset_scenario(self, run_manager, bot: MicroBot):
        await bot.clear_map()
        self.done = False
        self.winner = None

        # semi-random start location
        start_locations = sorted(bot.expansion_locations_list)
        bot.flag_location = start_locations[(run_manager.step_nr + (0 if bot == self.attacker_bot else random.randint(1, len(start_locations) - 1))) % len(start_locations)]

        # spawn flag
        await bot.client.debug_create_unit([[FLAG_BUILDING, 1, bot.flag_location, bot.player_id]])

        # asymmetrical scenario
        units_to_spawn = self.attacker_units if bot == self.attacker_bot else self.defender_units

        # spawn units
        for key in units_to_spawn.keys():
            await bot.spawn_army(UnitTypeId(key), units_to_spawn[key], bot.flag_location)

        await bot.reset_bot()
        await bot.say_name()

    async def check_if_done(self, run_manager) -> bool:
        return run_manager.episode_step_nr > self.settings.env_steps_per_episode or self.done


class PureCombatScenario(Scenario):
    """
    Simple 1 vs 1 combat scenario, last man standing wins.
    """

    def __init__(self, scenario_settings, attacker_bot: MicroBot, defender_bot: MicroBot, attacker_units: dict = None, defender_units: dict = None):
        super().__init__(scenario_settings=scenario_settings)
        self.name = f"PureCombatScenario"
        self.done = False
        self.attacker_bot = attacker_bot
        self.defender_bot = defender_bot

        if attacker_units is None:
            attacker_units = {UnitTypeId.MARINE: 9}
        if defender_units is None:
            defender_units = {UnitTypeId.ROACH: 4}

        self.attacker_units = attacker_units
        self.defender_units = defender_units

        self.reward_function = pure_combat_reward

    async def get_reward(self, run_manager, bot: MicroBot) -> float:
        reward, win_reward, winner = self.reward_function(run_manager, bot, self)
        if abs(win_reward) > 0:
            self.done = True
            self.winner = winner.name
        return reward + win_reward

    async def reset_scenario(self, data_collector, bot: MicroBot):
        await bot.clear_map()
        self.done = False
        self.winner = None

        # semi-random start location
        start_locations = sorted(bot.expansion_locations_list)
        bot.flag_location = start_locations[(data_collector.step_nr + (0 if bot == self.attacker_bot else random.randint(1, len(start_locations) - 1))) % len(start_locations)]

        # asymmetrical scenario
        units_to_spawn = self.attacker_units if bot == self.attacker_bot else self.defender_units

        # spawn units
        for key in units_to_spawn.keys():
            await bot.spawn_army(UnitTypeId(key), units_to_spawn[key], bot.flag_location)

        await bot.reset_bot()
        await bot.say_name()

    async def check_if_done(self, data_collector) -> bool:
        return data_collector.episode_step_nr > self.settings.env_steps_per_episode or self.done

