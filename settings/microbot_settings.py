from dataclasses import dataclass
from typing import List
from BattleML_definitions.actions_enum import ActionsML

from torch.utils.tensorboard import SummaryWriter


@dataclass
class MicroBotSettings:
    """
    Settings related to the input/output of a MicroBot
    """
    location_action_space_resolution_x: int
    location_action_space_resolution_y: int
    max_units: int

    mask_actions: List[ActionsML]

    def log(self, writer: SummaryWriter):
        writer.add_scalar("Info/spatial_output_resolution_x", self.location_action_space_resolution_x)
        writer.add_scalar("Info/spatial_output_resolution_y", self.location_action_space_resolution_y)
        writer.add_scalar("Info/max_units", self.max_units)

        with open(f"{writer.log_dir}/run_info.txt", "a") as f:
            f.write(f"location_action_space_resolution_x: {self.location_action_space_resolution_x}\n")
            f.write(f"location_action_space_resolution_y: {self.location_action_space_resolution_y}\n")
            f.write(f"max_units: {self.max_units}\n")

            f.write(f"masked actions: ")
            for action in self.mask_actions:
                f.write(f"{action.name}, ")
            f.write("\n")
