from sc2.position import Point2

from SC2BattleML.BattleML_definitions.actions_enum import ActionsML
from SC2BattleML.BattleML_definitions.observations import UNIT_ENCODING_KEY
from SC2BattleML.BattleML_definitions.unit_enum import UNITS_ML_DICT, FLAG_BUILDING
from SC2BattleML.bots.micro_bot import MicroBot
import numpy as np
from SC2BattleML.settings import MicroBotSettings


class AMoveBot(MicroBot):
    def __init__(self, name_postfix, settings: MicroBotSettings, observer=False):
        super().__init__(name_postfix=name_postfix, settings=settings, observer=observer)
        self.name = "A_MOVE_BOT" + name_postfix
        self.bot_type = "SCRIPTED"

    async def execute(self):
        observation = await self.get_observation()
        own_units, enemy_units, spatial_features, scalar_features, tags = observation

        nr_enemies = scalar_features[1]

        target_position = None
        for unit in enemy_units:
            if unit[UNITS_ML_DICT[FLAG_BUILDING.value]]:
                target_position = Point2((unit[UNIT_ENCODING_KEY["x_pos"]]*255, unit[UNIT_ENCODING_KEY["y_pos"]]*255))
                break

        if target_position is None:
            summed_position = Point2((0, 0))
            if nr_enemies > 0:
                for i in range(nr_enemies):
                    unit = enemy_units[i]
                    summed_position += Point2((unit[UNIT_ENCODING_KEY["x_pos"]]*255, unit[UNIT_ENCODING_KEY["y_pos"]]*255))
                target_position = summed_position / nr_enemies
            else:
                target_position = self.game_info.map_center

        location = self.inverse_transform_location(target_position)

        # always attack move all units
        actions = np.ones(len(own_units)) * ActionsML.ATTACK_MOVE.value

        # set attack location
        locations = np.ones(len(own_units))*location

        # target is never used
        targets = np.random.randint(0, self.settings.max_units, len(own_units))

        await self.execute_actions(actions, locations, targets, tags)
        return (actions, locations, targets), observation

    async def on_step(self, iteration: int):
        await self.data_collector.on_step(iteration, self)





