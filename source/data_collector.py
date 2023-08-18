from typing import List
import torch
from sc2.data import Result
from sc2.player import Bot, AbstractPlayer

from SC2BattleML.BattleML_definitions.actions_enum import ActionsML
from SC2BattleML.bots.micro_bot import MicroBot
from SC2BattleML.source.episode_data import Episode
from SC2BattleML.source.learning_manager import LearningManager
from SC2BattleML.scenarios.scenario import Scenario

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataCollector:
    def __init__(self, learning_manager: LearningManager, bots: List[MicroBot], scenario: Scenario, thread_nr: int):
        self.thread_nr = thread_nr
        self.learning_manager = learning_manager
        self.bots = bots
        for bot in bots:
            bot.data_collector = self

        self.scenario = scenario
        self.reward_function = self.scenario.get_reward

        self.episode = 0
        self.in_episode = False
        self.episode_step_nr = 0
        self.wait_game_steps = 3
        self.in_game_time: float = 0

        self.step_nr = 0
        self.step_nrs = {bot.name: -1 for bot in bots}
        self.start_time = 0
        self.current_episode = {bot.name: Episode() for bot in bots}

        self.prev_output = {bot.name: None for bot in bots}

        # load starting models from learning manager
        self.retrieve_model()
        self.learning_manager.increment_game_nr()
        print("thread nr:", thread_nr)

    async def on_step(self, step_nr: int, bot: MicroBot):
        self.in_game_time = bot.time

        self.step_nrs[bot.name] = step_nr
        all_bots_stepped = False
        if max(self.step_nrs.values()) == min(self.step_nrs.values()):
            all_bots_stepped = True

        if step_nr == 1 and all_bots_stepped and not self.scenario.minigame_map:
            self.reset_for_new_episode(commit_data=False)
            self.learning_manager.increment_game_nr()

        if step_nr < 5 and not self.scenario.minigame_map:
            bot.preserve_self()
            return

        if self.wait_game_steps > 0:
            if all_bots_stepped:
                self.wait_game_steps -= 1
            return


        if not self.in_episode:
            await self.start_episode(bot)

            if all_bots_stepped:
                self.episode += 1
                self.in_episode = True
                self.episode_step_nr = 0
                self.wait_game_steps = 3
            return

        if step_nr % self.scenario.settings.game_steps_per_env_step == 0:
            if all_bots_stepped:
                self.episode_step_nr += 1

            reward = await self.reward_function(self, bot)
            done = await self.scenario.check_if_done(self)
            if done:
                for micro_bot in self.bots:
                    self.current_episode[micro_bot.name].winner = self.scenario.winner

            if self.prev_output[bot.name] is not None and bot.bot_type == "RL":
                self.collect_trajectory(bot, self.prev_output[bot.name], reward, done)

            if done and all_bots_stepped:
                self.reset_for_new_episode()
                return

            if not bot.observer:
                bot_output = await bot.execute()
                self.prev_output[bot.name] = bot_output

    async def on_end(self, game_result: Result, bot):

        self.learning_manager.in_game_time_total[bot.name] += self.in_game_time

        if self.scenario.minigame_map:
            self.scenario.done = True

            await self.on_step(self.scenario.settings.game_steps_per_env_step*1000, bot)

            if game_result == Result.Victory:
                for micro_bot in self.bots:
                    self.current_episode[micro_bot.name].winner = bot.name

    async def start_episode(self, bot):
        await self.scenario.reset_scenario(self, bot)
        self.prev_output[bot.name] = None

    def reset_for_new_episode(self, commit_data=True):
        if commit_data:
            self.commit_data()
        self.retrieve_model()
        self.in_episode = False
        self.episode_step_nr = 0
        self.wait_game_steps = 3
        self.step_nr = 0
        self.step_nrs = {bot.name: 0 for bot in self.bots}

        self.prev_output = {bot.name: None for bot in self.bots}
        self.current_episode = {bot.name: Episode() for bot in self.bots}

    def collect_trajectory(self, bot: MicroBot, bot_output, reward, done):
        (actions, locations, targets), observation, value, lstm_state, (action_logits, location_logits, target_logits) = bot_output

        self.current_episode[bot.name].trajectory["action"].append(actions)
        self.current_episode[bot.name].trajectory["location"].append(locations)
        self.current_episode[bot.name].trajectory["target"].append(targets)

        self.current_episode[bot.name].trajectory["value"].append(value)
        self.current_episode[bot.name].trajectory["lstm_state"].append(lstm_state)
        self.current_episode[bot.name].trajectory["observation"].append(observation)
        self.current_episode[bot.name].trajectory["reward"].append(reward)
        self.current_episode[bot.name].trajectory["done"].append(done)

        self.current_episode[bot.name].trajectory["old_action_logit"].append(action_logits)
        self.current_episode[bot.name].trajectory["old_location_logit"].append(location_logits)
        self.current_episode[bot.name].trajectory["old_target_logit"].append(target_logits)

    def commit_data(self):
        with self.learning_manager.data_lock:
            print("commiting data, thread", self.thread_nr)
            for bot in self.bots:
                if bot.bot_type == "RL":
                    self.learning_manager.add_data(bot, self.current_episode[bot.name])

                # logging episode results
                won = 1 if self.current_episode[bot.name].winner == bot.name else 0
                tied = 1 if self.current_episode[bot.name].winner is None else 0
                loss = 1 if (self.current_episode[bot.name].winner and self.current_episode[bot.name].winner != bot.name) else 0

                self.learning_manager.wins[bot.name] += won
                self.learning_manager.ties[bot.name] += tied
                self.learning_manager.losses[bot.name] += loss

                self.learning_manager.last_results[bot.name].append(won)

                # logging action distribution results
                for action in ActionsML:
                    self.learning_manager.action_distribution[bot.name][action] += bot.action_log[action]
                bot.reset_action_log()


        self.learning_manager.received_data()

    def retrieve_model(self):
        with self.learning_manager.model_lock:
            for bot in [bot for bot in self.bots if bot.bot_type == "RL"]:
                print("retrieving state dict")
                bot.model.load_state_dict(self.learning_manager.state_dicts[bot.name])

    def get_players(self) -> List[AbstractPlayer]:
        return [Bot(bot.race, bot) for bot in self.bots]

