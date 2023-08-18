from __future__ import annotations
from abc import abstractmethod

from sc2.data import Race
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

from BattleML_definitions.actions_enum import ActionsML
from BattleML_definitions.observations import raw_units_processing, raw_image_processing
from BattleML_definitions.unit_enum import FLAG_BUILDING, PRESERVATION_BUILDINGS
from typing import List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from source.data_collector import DataCollector


class MicroBot(BotAI):
    def __init__(self, name_postfix, settings, observer=False, race=Race.Terran):
        super().__init__()
        self.data_collector: DataCollector = None
        self.flag_location: Point2 = None
        self.settings = settings

        self.name = "DEFAULT_NAME" + name_postfix
        self.bot_type = "DEFAULT_TYPE"
        self.observer = observer
        self.race = race

        self.action_log: dict[ActionsML, int] = {action: 0 for action in ActionsML}

    def reset_action_log(self):
        self.action_log = {action: 0 for action in ActionsML}

    async def spawn_army(self, unit_id: UnitTypeId, amount: int, location: Point2):
        await self.client.debug_create_unit([[unit_id, amount, location, self.player_id]])

    async def create_base(self, location: Point2):
        for unit in self.all_units:
            if unit.type_id == FLAG_BUILDING:
                await self.client.debug_kill_unit({unit.tag})

        await self.client.debug_create_unit([[FLAG_BUILDING, 1, location, self.player_id]])

    def get_flag(self) -> List[Unit]:
        return [unit for unit in self.all_own_units if unit.type_id == FLAG_BUILDING]

    def get_units(self) -> List[Unit]:
        return [unit for unit in self.all_own_units if not unit.is_structure]

    async def reset_bot(self):
        pass

    async def clear_map(self):
        for unit in self.all_own_units:
            if not unit.is_structure:
                await self.client.debug_kill_unit({unit.tag})
            if unit.type_id not in PRESERVATION_BUILDINGS:
                await self.client.debug_kill_unit({unit.tag})

    def preserve_self(self):
        for unit in self.all_own_units:
            if unit.type_id in PRESERVATION_BUILDINGS:
                unit(AbilityId.LIFT_COMMANDCENTER)
                unit.move(self.start_location.towards(self.game_info.map_center, -20), queue=True)

    async def get_observation(self):
        resp = await self.client.observation()
        # full observation data
        observation = resp.observation.observation

        # minimap feature layers (height_map, visibility_map, creep, camera, player_id, player_relative, selected,
        # alerts, buildable, pathable)
        spatial_obs = observation.feature_layer_data.minimap_renders

        # raw data (player, units, map_state, event, effects, radar)
        raw_obs = observation.raw_data

        own_units, enemy_units, scalar_features, tags = raw_units_processing(raw_obs, max_units=self.settings.max_units)

        spatial_features = raw_image_processing(spatial_obs)

        return [own_units, enemy_units, spatial_features, scalar_features, tags]

    def transform_location(self, location: int) -> Point2:
        resolution_x = self.settings.location_action_space_resolution_x
        resolution_y = self.settings.location_action_space_resolution_y

        loc_x = ((location % resolution_x) / resolution_x) * self.game_info.map_size[0]
        loc_y = ((resolution_y - (location // resolution_x)) / resolution_y) * self.game_info.map_size[1]
        return Point2((loc_x, loc_y))

    def inverse_transform_location(self, location: Point2) -> int:
        resolution_x = self.settings.location_action_space_resolution_x
        resolution_y = self.settings.location_action_space_resolution_y

        loc_x = int((location.x / self.game_info.map_size[0]) * resolution_x)
        loc_y = int(resolution_y - ((location.y / self.game_info.map_size[1]) * resolution_y)) * resolution_x

        return loc_x + loc_y

    async def execute_actions(self, actions: List[int], locations: List[int], targets: List[int], tags: tuple):
        unit_by_tag = {unit.tag: unit for unit in self.all_units}
        own_tags = tags[0]
        enemy_tags = tags[1]

        for i, unit_tag in enumerate(own_tags):
            self.action_log[ActionsML(actions[i])] += 1
            unit = unit_by_tag.get(unit_tag)

            # simple move
            if ActionsML(actions[i]) == ActionsML.MOVE:
                unit.move(self.transform_location(locations[i]))

            # hold position
            if ActionsML(actions[i]) == ActionsML.HOLD_POSITION:
                unit.hold_position()

            # attack move
            if ActionsML(actions[i]) == ActionsML.ATTACK_MOVE:
                unit.attack(self.transform_location(locations[i]))

            # attack unit
            if ActionsML(actions[i]) == ActionsML.ATTACK_TARGET:
                if targets[i] < len(enemy_tags):
                    enemy_unit = unit_by_tag.get(enemy_tags[targets[i]])
                    unit.attack(enemy_unit)

            # no operation; don't give any command
            if ActionsML(actions[i]) == ActionsML.NO_OPERATION:
                pass


    async def say_name(self):
        await self.client.chat_send(f"Bot Name: {self.name}", team_only=False)

    @abstractmethod
    async def execute(self):
        pass

