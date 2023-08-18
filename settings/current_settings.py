from sc2.ids.unit_typeid import UnitTypeId

from BattleML_definitions.actions_enum import ActionsML
from scenarios.minigame_scenario import MiniGameScenario
from settings import LearningSettings, ModelSettings, ScenarioSettings, HardwareSettings, MicroBotSettings


# resolution of spatial features we receive from sc2 client
# this number is not safe to change, SpatialEncoder and LocationHead will need to changed to match dimensions again
SPATIAL_RESOLUTION_X = 64
SPATIAL_RESOLUTION_Y = 64

learning_settings = LearningSettings(
    learning_rate_start=0.0001,
    learning_rate_decay=0.9992,
    minimum_learning_rate=0.00005,

    time_between_optimizations=10,

    gamma=0.99,

    eps_clip_start=0.2,
    eps_clip_decay=0.9992,
    eps_clip_min=0.02,

    gradient_clip=1,

    memory_size=4096,
    batch_size=512,
    epochs=3,

    loss_logging_frequency=10,
    reward_logging_frequency=20,
    model_save_frequency=10,

    evaluation_mode=False
)

model_settings = ModelSettings(
    unit_embedding_size=128,
    unit_transformer_feedforward_size=128,
    unit_transformer_nheads=2,
    unit_transformer_layers=2,

    spatial_embedding_size=128,
    map_skip_channels=16,

    core_layers=1,
    core_output_size=256,

    autoregressive_embedding_channels=4,
    target_head_attention_size=64,

    dropout=0,

    load_model=None
    )

microbot_settings = MicroBotSettings(
    location_action_space_resolution_x=62,
    location_action_space_resolution_y=62,
    max_units=14,
    mask_actions=[],
)

hardware_settings = HardwareSettings(
    max_game_time_limit=2*60*60,
    wait_between_game_launches=15,
    nr_simultaneous_games=4,
    max_nr_games=200000,
    realtime=False
)

# scenario_settings = ScenarioSettings(
#     scenario=PureCombatScenario,
#     minigame_scenario=False,
#     scenario_kwargs={"attacker_units": {UnitTypeId.MARINE: 5}, "defender_units": {UnitTypeId.MARINE: 5}},
#     map_name="Flat48",
#
#     game_steps_per_env_step=4,
#     env_steps_per_episode=80,
#
#     disable_fog=True
# )

scenario_settings = ScenarioSettings(
    scenario=MiniGameScenario,
    minigame_scenario=True,
    scenario_kwargs={},
    map_name="DefeatRoaches",

    game_steps_per_env_step=4,
    env_steps_per_episode=400,

    disable_fog=False
)
#
# scenario_settings = ScenarioSettings(
#     scenario=MiniGameScenario,
#     minigame_scenario=True,
#     scenario_kwargs={},
#     map_name="FindAndDefeatZerglings",
#
#     game_steps_per_env_step=4,
#     env_steps_per_episode=400,
#
#     disable_fog=False
# )
