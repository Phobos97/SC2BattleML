from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LearningSettings:
    """
    Settings related to how the models is trained
    """
    learning_rate_start: float
    learning_rate_decay: float
    minimum_learning_rate: float

    time_between_optimizations: float

    gamma: float

    eps_clip_start: float
    eps_clip_decay: float
    eps_clip_min: float

    gradient_clip: float

    memory_size: int
    batch_size: int
    epochs: int

    # settings related to logging frequency
    loss_logging_frequency: int
    reward_logging_frequency: int
    model_save_frequency: int

    evaluation_mode: bool
    
    def log(self, writer: SummaryWriter):
        writer.add_scalar("Info/learning_rate", self.learning_rate_start)
        writer.add_scalar("Info/learning_rate_decay", self.learning_rate_decay)
        writer.add_scalar("Info/learning_rate_minimum", self.minimum_learning_rate)

        writer.add_scalar("Info/time_between_optimizations", self.time_between_optimizations)

        writer.add_scalar("Info/gamma", self.gamma)

        writer.add_scalar("Info/eps_clip", self.eps_clip_start)
        writer.add_scalar("Info/eps_clip_decay", self.eps_clip_decay)
        writer.add_scalar("Info/eps_clip_min", self.eps_clip_min)

        writer.add_scalar("Info/gradient_clip", self.gradient_clip)

        writer.add_scalar("Info/memory_size", self.memory_size)
        writer.add_scalar("Info/batch_size", self.batch_size)
        writer.add_scalar("Info/epochs", self.epochs)

        writer.add_scalar("Info/loss_logging_frequency", self.loss_logging_frequency)
        writer.add_scalar("Info/reward_logging_frequency", self.reward_logging_frequency)
        writer.add_scalar("Info/model_save_frequency", self.model_save_frequency)

        writer.add_scalar("Info/evaluation_mode", int(self.evaluation_mode))

        with open(f"{writer.log_dir}/run_info.txt", "a") as f:
            f.write(f"learning_rate_start: {self.learning_rate_start}\n")
            f.write(f"learning_rate_decay: {self.learning_rate_decay}\n")
            f.write(f"minimum_learning_rate: {self.minimum_learning_rate}\n")
            f.write(f"time_between_optimizations: {self.time_between_optimizations}\n")

            f.write(f"gamma: {self.gamma}\n")

            f.write(f"eps_clip: {self.eps_clip_start}\n")
            f.write(f"eps_clip_decay: {self.eps_clip_decay}\n")
            f.write(f"eps_clip_min: {self.eps_clip_min}\n")

            f.write(f"gradient_clip: {self.gradient_clip}\n")

            f.write(f"memory_size: {self.memory_size}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"epochs: {self.epochs}\n")

            f.write(f"loss_logging_frequency: {self.loss_logging_frequency}\n")
            f.write(f"reward_logging_frequency: {self.reward_logging_frequency}\n")
            f.write(f"model_save_frequency: {self.model_save_frequency}\n")

            f.write(f"evaluation_mode: {self.evaluation_mode}\n")
