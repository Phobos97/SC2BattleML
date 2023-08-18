from dataclasses import dataclass
from typing import Union

from torch.utils.tensorboard import SummaryWriter


@dataclass
class ModelSettings:
    """
    Settings related to the architecture of the BattleML models
    """
    unit_embedding_size: int
    unit_transformer_feedforward_size: int
    unit_transformer_nheads: int
    unit_transformer_layers: int

    spatial_embedding_size: int
    map_skip_channels: int

    core_layers: int
    core_output_size: int

    autoregressive_embedding_channels: int
    target_head_attention_size: int

    dropout: float

    load_model: Union[None, str]

    def log(self, writer: SummaryWriter):
        writer.add_scalar("Info/unit_embedding_size", self.unit_embedding_size)
        writer.add_scalar("Info/unit_transformer_feedforward_size", self.unit_transformer_feedforward_size)
        writer.add_scalar("Info/unit_transformer_nheads", self.unit_transformer_nheads)
        writer.add_scalar("Info/unit_transformer_layers", self.unit_transformer_layers)

        writer.add_scalar("Info/spatial_embedding_size", self.spatial_embedding_size)
        writer.add_scalar("Info/map_skip_channels", self.map_skip_channels)

        writer.add_scalar("Info/core_layers", self.core_layers)
        writer.add_scalar("Info/core_output_size", self.core_output_size)
        writer.add_scalar("Info/autoregressive_embedding_channels", self.autoregressive_embedding_channels)
        writer.add_scalar("Info/target_head_attention_size", self.target_head_attention_size)

        writer.add_scalar("Info/dropout", self.dropout)

        writer.add_scalar("Info/load_model", int(bool(self.load_model)))

        with open(f"{writer.log_dir}/run_info.txt", "a") as f:
            f.write(f"unit_embedding_size: {self.unit_embedding_size}\n")
            f.write(f"unit_transformer_feedforward_size: {self.unit_transformer_feedforward_size}\n")
            f.write(f"unit_transformer_nheads: {self.unit_transformer_nheads}\n")
            f.write(f"unit_transformer_layers: {self.unit_transformer_layers}\n")

            f.write(f"spatial_embedding_size: {self.spatial_embedding_size}\n")
            f.write(f"map_skip_channels: {self.map_skip_channels}\n")

            f.write(f"core_layers: {self.core_layers}\n")
            f.write(f"core_output_size: {self.core_output_size}\n")

            f.write(f"autoregressive_embedding_channels: {self.autoregressive_embedding_channels}\n")
            f.write(f"target_head_attention_size: {self.target_head_attention_size}\n")

            f.write(f"dropout: {self.dropout}\n")
            f.write(f"load_model: {self.load_model}\n")



