import asyncio

from sc2 import maps
from sc2.main import GameMatch, a_run_multiple_games

from bots import AMoveBot, DoNothingBot, HoldPositionBot, MLBot
from source import LearningManager, DataCollector
from settings import ModelSettings, LearningSettings, MicroBotSettings, HardwareSettings, ScenarioSettings


async def start_game_multi(model_settings: ModelSettings, learning_settings: LearningSettings,
                           bot_settings: MicroBotSettings, hardware_settings: HardwareSettings,
                           scenario_settings: ScenarioSettings, minigame_mode=True, run_name=""):
    # define the types of bots and scenario they will be playing
    bot1 = AMoveBot(name_postfix="_defender", observer=False, settings=bot_settings)
    bot2 = MLBot(name_postfix="_attacker", model_settings=model_settings, settings=bot_settings)

    # learning manager will be receiving environment data from workers and optimizing RL bot models
    learning_manager = LearningManager(bots=[bot1, bot2], run_name=f"{scenario_settings.scenario.__name__}_{scenario_settings.map_name}_{run_name}",
                                       nr_workers=hardware_settings.nr_simultaneous_games,
                                       learning_settings=learning_settings, scenario_settings=scenario_settings,
                                       hardware_settings=hardware_settings)


    # create a WorkerManager for each concurrent game,
    if scenario_settings.minigame_scenario:
        # minigame scenario's require just a single bot as apposed to 2 for non-minigames
        bots = [[MLBot(name_postfix="_attacker", model_settings=model_settings, settings=bot_settings)] for i in range(hardware_settings.nr_simultaneous_games)]
        workers = [DataCollector(learning_manager, bots=bots[i],
                                 scenario=scenario_settings.scenario(scenario_settings, **scenario_settings.scenario_kwargs),
                                 thread_nr=i) for i in range(hardware_settings.nr_simultaneous_games)]
    else:
        bots = [[AMoveBot(name_postfix="_defender", settings=bot_settings), MLBot(name_postfix="_attacker", model_settings=model_settings, settings=bot_settings)] for i in range(hardware_settings.nr_simultaneous_games)]
        workers = [DataCollector(learning_manager, bots=bots[i],
                                 scenario=scenario_settings.scenario(scenario_settings, attacker_bot=bots[i][1], defender_bot=bots[i][0], **scenario_settings.scenario_kwargs),
                                 thread_nr=i) for i in range(hardware_settings.nr_simultaneous_games)]



    map = maps.get(scenario_settings.map_name)

    # create GameMatch for each concurrent game
    matches: [GameMatch] = [GameMatch(map_sc2=map, players=workers[i].get_players(),
                                      realtime=hardware_settings.realtime, disable_fog=scenario_settings.disable_fog,
                                      game_time_limit=hardware_settings.max_game_time_limit)
                            for i in range(hardware_settings.nr_simultaneous_games)]


    run_game_functions = (a_run_multiple_games([match]*hardware_settings.max_nr_games) for match in matches)

    tasks = []
    for game_func in run_game_functions:
        tasks.append(asyncio.create_task(game_func))
        # wait some seconds between client launches to prevent crashes
        await asyncio.sleep(hardware_settings.wait_between_game_launches)


    loop = asyncio.get_event_loop()
    training_loop = loop.create_task(learning_manager.training_loop())
    await asyncio.gather(*tasks)

    loop.call_later(10, training_loop.cancel)
    try:
        await asyncio.gather(training_loop)
    except asyncio.CancelledError:
        pass

    print("all threads done!")

