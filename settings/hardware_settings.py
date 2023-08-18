from dataclasses import dataclass
import multiprocessing
import torch
import platform
from torch.utils.tensorboard import SummaryWriter


@dataclass
class HardwareSettings:
    max_game_time_limit: int
    wait_between_game_launches: float
    nr_simultaneous_games: int
    max_nr_games: int
    realtime: bool

    cpu_cores = multiprocessing.cpu_count()
    gpu_model = torch.cuda.get_device_name()
    cpu_model = platform.processor()

    def log(self, writer: SummaryWriter):
        writer.add_scalar("Info/max_game_time_limit", self.max_game_time_limit)
        writer.add_scalar("Info/nr_simultaneous_games", self.nr_simultaneous_games)
        writer.add_scalar("Info/cpu_cores", self.cpu_cores)

        with open(f"{writer.log_dir}/run_info.txt", "a") as f:
            f.write(f"max_game_time_limit: {self.max_game_time_limit}\n")
            f.write(f"wait_between_game_launches: {self.wait_between_game_launches}\n")
            f.write(f"nr_simultaneous_games: {self.nr_simultaneous_games}\n")
            f.write(f"max_nr_games: {self.max_nr_games}\n")
            f.write(f"realtime: {self.realtime}\n")

            f.write(f"cpu_cores: {self.cpu_cores}\n")
            f.write(f"gpu_model: {self.gpu_model}\n")
            f.write(f"cpu_model: {self.cpu_model}\n")

