from __future__ import annotations
from scenarios.scenario import Scenario
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bots.micro_bot import MicroBot
    from settings import ScenarioSettings


class MiniGameScenario(Scenario):
    """
    Used for running the Blizzard-made minigame maps.
    """

    def __init__(self, scenario_settings: ScenarioSettings):
        super().__init__(scenario_settings=scenario_settings)
        self.name = f"MiniGame_{scenario_settings.map_name}"
        self.done = False
        self.minigame_map = True
        self.previous_score = {}

    async def get_reward(self, run_manager, bot: MicroBot) -> float:
        if bot.name not in self.previous_score.keys():
            self.previous_score[bot.name] = 0

        resp = await bot.client.observation()
        observation = resp.observation.observation
        score = observation.score.score
        delta_score = score - self.previous_score[bot.name]

        self.previous_score[bot.name] = score
        return delta_score

    async def reset_scenario(self, run_manager, bot: MicroBot):
        self.done = False
        self.winner = None
        self.previous_score = {}

    async def check_if_done(self, run_manager) -> bool:
        return self.done


