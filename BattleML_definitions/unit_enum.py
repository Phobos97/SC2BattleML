# reduced list from sc2.ids.UnitTypeId
# only listing relevant multiplayer units (not buildings)

import enum
from sc2.ids.unit_typeid import UnitTypeId

FLAG_BUILDING = UnitTypeId.SUPPLYDEPOT
PRESERVATION_BUILDINGS = [UnitTypeId.COMMANDCENTER, UnitTypeId.COMMANDCENTERFLYING]
PRESERVATION_BUILDING_IDS = [building.value for building in PRESERVATION_BUILDINGS]

FLAG_ID = FLAG_BUILDING.value

class UnitsML(enum.Enum):
    FLAG = FLAG_ID

    NOTAUNIT = 0
    SYSTEM_SNAPSHOT_DUMMY = 1

    # COLOSSUS = 4
    # INFESTORTERRAN = 7
    # BANELINGCOCOON = 8
    BANELING = 9
    # MOTHERSHIP = 10

    # SIEGETANKSIEGED = 32
    # SIEGETANK = 33
    # VIKINGASSAULT = 34
    # VIKINGFIGHTER = 35

    # SCV = 45
    MARINE = 48
    REAPER = 49
    # GHOST = 50
    MARAUDER = 51
    # THOR = 52
    # HELLION = 53
    # MEDIVAC = 54
    # BANSHEE = 55
    # RAVEN = 56
    # BATTLECRUISER = 57
    # NUKE = 58

    ZEALOT = 73
    STALKER = 74
    # HIGHTEMPLAR = 75
    # DARKTEMPLAR = 76
    # SENTRY = 77
    # PHOENIX = 78
    # CARRIER = 79
    # VOIDRAY = 80
    # WARPPRISM = 81
    # OBSERVER = 82
    IMMORTAL = 83
    # PROBE = 84
    # INTERCEPTOR = 85

    # DRONE = 104
    ZERGLING = 105
    # OVERLORD = 106
    HYDRALISK = 107
    # MUTALISK = 108
    # ULTRALISK = 109
    ROACH = 110
    # INFESTOR = 111
    # CORRUPTOR = 112
    # BROODLORDCOCOON = 113
    # BROODLORD = 114
    # QUEEN = 126
    # OVERSEER = 129

    # ARCHON = 141
    # ADEPT = 311

    # SWARMHOSTMP = 494
    # ORACLE = 495
    # TEMPEST = 496

    # WIDOWMINE = 498
    # VIPER = 499
    # WIDOWMINEBURROWED = 500

    # LURKERMP = 502
    # LURKERMPBURROWED = 503

    # RAVAGER = 688
    # LIBERATOR = 689
    # THORAP = 691
    # CYCLONE = 692
    # LOCUSTMPFLYING = 693
    # DISRUPTOR = 694

    # LIBERATORAG = 734

    # ADEPTPHASESHIFT = 801

    # LURKER = 911
    # LURKERBURROWED = 912

    # VIKING = 1940

    def __repr__(self) -> str:
        return f"UnitTypeId.{self.name}"


UNITS_ML_DICT = {}
for i, unit in enumerate(UnitsML):
    UNITS_ML_DICT[unit.value] = i


UNIT_TYPE_COUNT = len(UnitsML.__members__)

