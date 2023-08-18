from SC2BattleML.BattleML_definitions.actions_enum import ACTION_COUNT
from SC2BattleML.bots.micro_bot import MicroBot
import numpy as np
from SC2BattleML.settings import MicroBotSettings


class RandomBot(MicroBot):
    def __init__(self, name_postfix, settings: MicroBotSettings, observer=False):
        super().__init__(name_postfix=name_postfix, settings=settings, observer=observer)
        self.name = "RANDOM_BOT" + name_postfix
        self.bot_type = "SCRIPTED"

    async def execute(self):
        observation = await self.get_observation()
        own_units, enemy_units, spatial_features, scalar_features, tags = observation

        resolution_x = self.settings.location_action_space_resolution_x
        resolution_y = self.settings.location_action_space_resolution_y

        actions = np.random.randint(0, ACTION_COUNT, len(own_units))
        locations = np.random.randint(0, resolution_x * resolution_y, len(own_units))
        targets = np.random.randint(0, self.settings.max_units, len(own_units))

        await self.execute_actions(actions, locations, targets, tags)
        return (actions, locations, targets), observation

    async def on_step(self, iteration: int):
        await self.data_collector.on_step(iteration, self)











