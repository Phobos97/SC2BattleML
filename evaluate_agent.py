import asyncio

from sc2.ids.unit_typeid import UnitTypeId
from scenarios.combat_scenario import PureCombatScenario
from scenarios.minigame_scenario import MiniGameScenario
from settings import ScenarioSettings
from source.local_run import start_game_multi
import time

from settings.current_settings import model_settings, learning_settings, microbot_settings, hardware_settings

if __name__ == '__main__':
    # which models to evaluate
    load_model_path = "[PATH-TO-MODEL-CHECKPOINT]"
    model_settings.load_model = load_model_path

    # turn on evaluation mode to stop any training
    learning_settings.evaluation_mode = True

    # use realtime mode to spectate behaviour
    hardware_settings.realtime = False
    hardware_settings.nr_simultaneous_games = 4

    # evaluate for 250 games per nr_simultaneous_games (e.g. 4*250 = 1000 total episodes on minigames)
    hardware_settings.max_nr_games = 250

    # which scenario to evaluate on
    scenario_settings = ScenarioSettings(
        scenario=MiniGameScenario,
        minigame_scenario=True,
        scenario_kwargs={},
        map_name="DefeatRoaches",

        game_steps_per_env_step=4,
        env_steps_per_episode=400,

        disable_fog=False
    )

    
    start_time = time.time()
    asyncio.run(start_game_multi(model_settings=model_settings,
                                 learning_settings=learning_settings, bot_settings=microbot_settings,
                                 hardware_settings=hardware_settings, scenario_settings=scenario_settings,
                                 run_name="evaluation"))

    print("time taken to simulate game:", time.time() - start_time)

