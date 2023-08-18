from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bots.micro_bot import MicroBot

class Scenario:
    def __init__(self, scenario_settings):
        self.name = "default"
        self.done = False
        self.minigame_map = False
        self.settings = scenario_settings
        self.winner = None

    @abstractmethod
    async def get_reward(self, run_manager, bot: MicroBot):
        raise NotImplementedError

    @abstractmethod
    async def reset_scenario(self, run_manager, bot: MicroBot):
        raise NotImplementedError

    @abstractmethod
    async def check_if_done(self, run_manager):
        raise NotImplementedError



