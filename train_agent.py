import asyncio
from SC2BattleML.source.local_run import start_game_multi
import time

from SC2BattleML.settings.current_settings import model_settings, learning_settings, microbot_settings, hardware_settings, \
    scenario_settings

if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(start_game_multi(model_settings=model_settings,
                                 learning_settings=learning_settings, bot_settings=microbot_settings,
                                 hardware_settings=hardware_settings, scenario_settings=scenario_settings))

    print("time taken to simulate game:", time.time() - start_time)

