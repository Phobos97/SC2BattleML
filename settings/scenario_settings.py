from __future__ import annotations

from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from SC2BattleML.scenarios.scenario import Scenario

@dataclass
class ScenarioSettings:
    scenario: type[Scenario]
    minigame_scenario: bool
    scenario_kwargs: dict
    map_name: str

    game_steps_per_env_step: int
    env_steps_per_episode: int

    disable_fog: bool

    def log(self, writer: SummaryWriter):
        writer.add_scalar("Info/game_steps_per_env_step", self.game_steps_per_env_step)
        writer.add_scalar("Info/env_steps_per_episode", self.env_steps_per_episode)
        writer.add_scalar("Info/disable_fog", int(self.disable_fog))

        with open(f"{writer.log_dir}/run_info.txt", "a") as f:
            f.write(f"scenario_name: {self.scenario.__name__}\n")
            f.write(f"minigame_scenario: {self.minigame_scenario}\n")
            f.write(f"scenario_kwargs: {self.scenario_kwargs}\n")
            f.write(f"map: {self.map_name}\n")

            f.write(f"game_steps_per_env_step: {self.game_steps_per_env_step}\n")
            f.write(f"env_steps_per_episode: {self.env_steps_per_episode}\n")

            f.write(f"disable_fog: {self.disable_fog}\n")

