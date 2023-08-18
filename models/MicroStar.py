from torch import nn, Tensor
import torch
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
import torch.nn.functional as F

from BattleML_definitions.actions_enum import ACTION_COUNT
from BattleML_definitions.observations import UNIT_ENCODING_SIZE, SPACIAL_FEATURES, SCALAR_FEATURE_SIZE
from settings import ModelSettings, MicroBotSettings
from settings.current_settings import SPATIAL_RESOLUTION_X, SPATIAL_RESOLUTION_Y
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, settings: ModelSettings):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(settings.core_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, core_output: Tensor) -> Tensor:
        """
        :param core_output: [batch, 1, core_output_size]
        :return: [batch, 1, 1]
        """
        x = self.fc1(core_output)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.fc3(x)

        return value


class MicroStar(nn.Module):
    def __init__(self, model_settings: ModelSettings, bot_settings: MicroBotSettings):
        super(MicroStar, self).__init__()
        self.settings = model_settings
        self.bot_settings = bot_settings

        # encoders
        self.spatial_encoder = SpatialEncoder(self.settings)
        self.unit_encoder = UnitEncoder(self.settings, self.bot_settings.max_units)

        # core
        self.core = Core(self.settings)

        # heads
        self.action_head = ActionHead(self.settings, self.bot_settings.max_units, self.spatial_encoder.map_skip_dims)
        self.location_head = LocationHead(self.settings, self.bot_settings, self.spatial_encoder.map_skip_dims)
        self.target_head = TargetHead(self.settings, self.bot_settings.max_units, self.spatial_encoder.map_skip_dims)

        self.critic = ValueNetwork(self.settings)

    def forward(self, own_units: Tensor, enemy_units: Tensor, map_features: Tensor, scalar_features: Tensor,
                lstm_state: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        :param own_units: [batch, max_units, UNIT_ENCODING_SIZE]
        :param enemy_units: [batch, max_units, UNIT_ENCODING_SIZE]
        :param map_features: [batch, SPACIAL_FEATURES, SPATIAL_RESOLUTION_X, SPATIAL_RESOLUTION_Y]
        :param scalar_features: [batch, SCALAR_FEATURE_SIZE]
        :param lstm_state: ([core_layers, batch, core_output_size], [core_layers, batch, core_output_size])
        :return: [batch, max_units, ACTION_COUNT],
                 [batch, max_units, location_action_space_resolution_x, location_action_space_resolution_y],
                 [batch, max_units, max_units],
                 [batch, 1, 1],
                 ([core_layers, batch, core_output_size], [core_layers, batch, core_output_size])
        """
        nr_own_units = scalar_features[:, 0]
        nr_enemy_units = scalar_features[:, 1]
        nr_own_flags = scalar_features[:, 2]

        own_unit_embeddings, own_embedded_unit, enemy_unit_embeddings, enemy_embedded_unit =\
            self.unit_encoder(own_units, nr_own_units, enemy_units, nr_enemy_units)
        embedded_spatial, map_skip = self.spatial_encoder(map_features)

        core_output, new_lstm_state = self.core(own_embedded_unit, enemy_embedded_unit, embedded_spatial, scalar_features, lstm_state)

        # get output distributions
        action_logits, autoregressive_embedding = self.action_head(core_output, own_unit_embeddings, nr_own_units, nr_own_flags)
        location_logits = self.location_head(autoregressive_embedding, map_skip)
        target_logits = self.target_head(autoregressive_embedding, own_unit_embeddings, enemy_unit_embeddings, nr_own_units, nr_enemy_units, nr_own_flags)

        value = self.critic(core_output)
        return action_logits, location_logits, target_logits, value, new_lstm_state


class Core(nn.Module):
    def __init__(self, settings: ModelSettings):
        super(Core, self).__init__()

        core_input_size = settings.unit_embedding_size * 2 + settings.spatial_embedding_size + SCALAR_FEATURE_SIZE
        self.core = nn.LSTM(input_size=core_input_size, hidden_size=settings.core_output_size,
                            num_layers=settings.core_layers, batch_first=True)

        self.dropout = nn.Dropout(p=settings.dropout)

    def forward(self, own_embedded_unit: Tensor, enemy_embedded_unit: Tensor, embedded_spatial: Tensor,
                scalar_features: Tensor, lstm_state: Tensor=None) -> (Tensor, Tensor):
        """
        :param own_embedded_unit: [batch, 1, UNIT_ENCODING_SIZE]
        :param enemy_embedded_unit: [batch, 1, UNIT_ENCODING_SIZE]
        :param embedded_spatial: [batch, spatial_embedding_size]
        :param scalar_features: [batch, SCALAR_FEATURE_SIZE]
        :param lstm_state: ([core_layers, batch, core_output_size], [core_layers, batch, core_output_size])
        :return: [batch, 1, core_output_size],
                 ([core_layers, batch, core_output_size], [core_layers, batch, core_output_size])
        """
        core_input = torch.cat([own_embedded_unit, enemy_embedded_unit, embedded_spatial, scalar_features], dim=-1).unsqueeze(1)

        if lstm_state:
            core_output, new_lstm_state = self.core(core_input, lstm_state)
        else:
            core_output, new_lstm_state = self.core(core_input)

        core_output = self.dropout(core_output)

        return core_output, new_lstm_state


class UnitEncoder(nn.Module):
    def __init__(self, settings: ModelSettings, max_units: int):
        super(UnitEncoder, self).__init__()

        self.unit_embedding_size = settings.unit_embedding_size
        self.nheads = settings.unit_transformer_nheads
        self.max_units = max_units

        self.embedding_layer = nn.Linear(UNIT_ENCODING_SIZE, self.unit_embedding_size)

        encoder_layer = TransformerEncoderLayer(d_model=self.unit_embedding_size, nhead=self.nheads,
                                                dim_feedforward=settings.unit_transformer_feedforward_size, batch_first=True)
        encoder_norm = LayerNorm(self.unit_embedding_size)
        self.UnitEncoder = TransformerEncoder(encoder_layer, num_layers=settings.unit_transformer_layers, norm=encoder_norm)

    def forward(self, own_units: Tensor, nr_own_units: Tensor, enemy_units: Tensor, nr_enemy_units: Tensor)\
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        :param own_units: [batch, max_units, UNIT_ENCODING_SIZE]
        :param nr_own_units: [batch, 1]
        :param enemy_units: [batch, max_units, UNIT_ENCODING_SIZE]
        :param nr_enemy_units: [batch, 1]
        :return: [batch, max_units, UNIT_ENCODING_SIZE], [batch, 1, UNIT_ENCODING_SIZE],
                 [batch, max_units, UNIT_ENCODING_SIZE], [batch, 1, UNIT_ENCODING_SIZE]
        """
        own_units_embed = self.embedding_layer(own_units.float())
        enemy_units_embed = self.embedding_layer(enemy_units.float())

        all_units = torch.cat([own_units_embed, enemy_units_embed], dim=-2)

        batch_size = len(nr_own_units)
        mask = torch.ones((batch_size, self.max_units * 2), dtype=torch.bool).to(device)
        for i in range(batch_size):
            mask[i, :nr_own_units[i]] = False
            mask[i, self.max_units:self.max_units + nr_enemy_units[i]] = False
            if nr_own_units[i] == 0:
                mask[i, 0] = False

        unit_embeddings = self.UnitEncoder(all_units, src_key_padding_mask=mask)

        own_unit_embeddings = unit_embeddings[:, :self.max_units, :]
        own_embedded_unit = torch.mean(own_unit_embeddings, dim=1)

        enemy_unit_embeddings = unit_embeddings[:, self.max_units:, :]
        enemy_embedded_unit = torch.mean(own_unit_embeddings, dim=1)

        return own_unit_embeddings, own_embedded_unit, enemy_unit_embeddings, enemy_embedded_unit

def calc_post_conv_dims(input_shape: tuple, kernel_size: int, stride: int=1, padding: int=0) -> tuple:
    return tuple(round((test - kernel_size + 2 * padding) / stride + 1) for test in input_shape)


class SpatialEncoder(nn.Module):
    def __init__(self, settings: ModelSettings):
        super(SpatialEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=SPACIAL_FEATURES, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=settings.map_skip_channels, kernel_size=4, stride=2)

        new_dims1 = calc_post_conv_dims((SPATIAL_RESOLUTION_X, SPATIAL_RESOLUTION_Y), kernel_size=4, stride=2)
        self.map_skip_dims = calc_post_conv_dims(new_dims1, kernel_size=4, stride=2)

        new_size = settings.map_skip_channels*np.prod(self.map_skip_dims)
        self.fc = nn.Linear(new_size, settings.spatial_embedding_size)

    def forward(self, map_features: Tensor) -> (Tensor, Tensor):
        """
        :param map_features: [batch, SPACIAL_FEATURES, SPATIAL_RESOLUTION_X, SPATIAL_RESOLUTION_Y]
        :return: [batch, spatial_embedding_size], [batch, map_skip_channels, map_skip_dims[0], map_skip_dims[1]]
        """
        # downsample map
        spatial = self.conv1(map_features)
        spatial = F.relu(spatial)
        spatial = self.conv2(spatial)
        spatial = F.relu(spatial)
        spatial = self.conv3(spatial)
        map_skip = F.relu(spatial)

        embedded_spatial = self.fc(map_skip.flatten(start_dim=1))
        embedded_spatial = F.relu(embedded_spatial)

        return embedded_spatial, map_skip


class ActionHead(nn.Module):
    def __init__(self, settings: ModelSettings, max_units: int, map_skip_dims: tuple):
        super(ActionHead, self).__init__()

        self.hidden_layer_size = settings.autoregressive_embedding_channels * np.prod(map_skip_dims)
        self.max_units = max_units

        self.fc1 = nn.Linear(settings.core_output_size + settings.unit_embedding_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, ACTION_COUNT)

    def forward(self, core_output: Tensor, own_unit_embeddings: Tensor, nr_units: Tensor, nr_own_flags: Tensor)\
            -> (Tensor, Tensor):
        """
        :param core_output: [batch, 1, core_output_size]
        :param own_unit_embeddings: [batch, max_units, unit_embedding_size]
        :param nr_units: [batch, 1]
        :param nr_own_flags: [batch, 1]
        :return: [batch, max_units, ACTION_COUNT],
                 [batch, max_units, autoregressive_embedding_channels * np.prod(map_skip_dims)]
        """
        core_output = core_output.repeat(1, self.max_units, 1)

        combined_input = torch.cat([own_unit_embeddings, core_output], dim=-1)

        x = self.fc1(combined_input)
        autoregressive_embedding = F.relu(x)

        x = self.fc2(autoregressive_embedding)

        action_logits = F.softmax(x, dim=-1)

        # mask out actions of un-used units
        mask = torch.ones_like(action_logits) * 1e-9
        for i in range(len(nr_units)):
            mask[i, nr_own_flags[i]:nr_units[i], :] = 1

        action_logits = action_logits * mask

        return action_logits, autoregressive_embedding


class LocationHead(nn.Module):
    def __init__(self, settings: ModelSettings, bot_settings: MicroBotSettings, map_skip_dims: tuple):
        super(LocationHead, self).__init__()
        self.autoregressive_embedding_channels = settings.autoregressive_embedding_channels
        self.map_skip_dims = map_skip_dims
        self.input_channels = settings.map_skip_channels + settings.autoregressive_embedding_channels

        self.bot_settings = bot_settings

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2)


    def forward(self, autoregressive_embedding: Tensor, map_skip: Tensor) -> Tensor:
        """
        :param autoregressive_embedding: [batch, max_units, autoregressive_embedding_channels*prod(map_skip_dims)]
        :param map_skip: [batch, map_skip_dims[0], map_skip_dims[1]]
        :return: [batch, max_units, location_action_space_resolution_x, location_action_space_resolution_y]
        """
        autoregressive_embedding = autoregressive_embedding.reshape((-1, self.bot_settings.max_units,
                                                                     self.autoregressive_embedding_channels,
                                                                     self.map_skip_dims[-2], self.map_skip_dims[-1]))

        map_skip = map_skip.unsqueeze(1).repeat(1, self.bot_settings.max_units, 1, 1, 1)
        combined = torch.cat([autoregressive_embedding, map_skip], dim=2)

        WIDTH = combined.shape[-1]
        HEIGHT = combined.shape[-2]
        CHANNELS = combined.shape[-3]

        unbatched = combined.view(-1, CHANNELS, HEIGHT, WIDTH)

        x = self.conv1(unbatched)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # up-sample
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)

        x = x.view(-1, self.bot_settings.max_units, 1, x.shape[-2], x.shape[-1])

        assert x.shape[-2] == self.bot_settings.location_action_space_resolution_x
        assert x.shape[-1] == self.bot_settings.location_action_space_resolution_y

        flattened = x.flatten(start_dim=2)
        location_logits = F.softmax(flattened, dim=-1)

        return location_logits


class TargetHead(nn.Module):
    def __init__(self, settings: ModelSettings, max_units: int, map_skip_dims: tuple):
        super(TargetHead, self).__init__()
        self.max_units = max_units
        self.autoregressive_embedding_size = settings.autoregressive_embedding_channels * np.prod(map_skip_dims)
        self.attention_size = settings.target_head_attention_size

        self.query = nn.Linear(settings.unit_embedding_size + self.autoregressive_embedding_size, settings.target_head_attention_size)
        self.key = nn.Linear(settings.unit_embedding_size, settings.target_head_attention_size)

    def forward(self, autoregressive_embedding: Tensor, own_unit_embeddings: Tensor, enemy_unit_embeddings: Tensor,
                nr_own_units: Tensor, nr_units_enemy: Tensor, nr_own_flags: Tensor) -> Tensor:
        """
        :param autoregressive_embedding: [batch, max_units, autoregressive_embedding_size]
        :param own_unit_embeddings: [batch, max_units, unit_embedding_size]
        :param enemy_unit_embeddings: [batch, max_units, unit_embedding_size]
        :param nr_own_units: [batch, 1]
        :param nr_units_enemy: [batch, 1]
        :param nr_own_flags: [batch, 1]
        :return: [batch, max_units, max_units]
        """
        combined = torch.cat([autoregressive_embedding, own_unit_embeddings], dim=2)

        queries = self.query(combined)
        keys = self.key(enemy_unit_embeddings)

        attention = torch.div(torch.bmm(queries, keys.transpose(1, 2)), self.attention_size ** 0.5)

        # mask out invalid targets
        batch_size = attention.shape[0]
        mask = torch.ones((batch_size, self.max_units, self.max_units), device=device) * 1e-9
        for i in range(batch_size):
            mask[i, nr_own_flags[i]:nr_own_units[i], 0:nr_units_enemy[i]] = 1

        attention = torch.mul(attention, mask)
        target_logits = F.softmax(attention, dim=-1)

        return target_logits




