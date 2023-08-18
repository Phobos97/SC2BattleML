import asyncio
from sc2 import maps
from sc2.data import Race
from sc2.main import GameMatch, a_run_multiple_games
from sc2.player import Human

if __name__ == '__main__':
    # simple script to play Blizzard provided minigames yourself :)

    # map_name = "DefeatRoaches"
    map_name = "DefeatZerglingsAndBanelings"
    # map_name = "FindAndDefeatZerglings"

    map = maps.get(map_name)
    repeat_n_times = 3

    matches: [GameMatch] = [GameMatch(map_sc2=map, players=[Human(race=Race.Terran)],
                                      realtime=True, disable_fog=False)]*repeat_n_times

    asyncio.run(a_run_multiple_games(matches))

