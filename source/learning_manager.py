import asyncio
import time
import datetime
import os
import numpy as np
import random
from typing import List
import torch
from threading import Lock
from colorama import init as colorama_init
from colorama import Fore, Style
import re

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from SC2BattleML.BattleML_definitions.actions_enum import ActionsML
from SC2BattleML.BattleML_definitions.observations import SPACIAL_FEATURES
from SC2BattleML.bots.micro_bot import MicroBot
from SC2BattleML.source.episode_data import Episode
from SC2BattleML.settings import LearningSettings, ScenarioSettings, HardwareSettings
from SC2BattleML.settings.current_settings import SPATIAL_RESOLUTION_X, SPATIAL_RESOLUTION_Y

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
colorama_init()


class LearningManager:
    def __init__(self, bots: List[MicroBot], run_name: str, nr_workers: int, learning_settings: LearningSettings,
                 scenario_settings: ScenarioSettings, hardware_settings: HardwareSettings):
        # learning parameters
        self.settings = learning_settings
        self.scenario_settings = scenario_settings
        self.hardware_settings = hardware_settings

        self.nr_workers = nr_workers

        self.episode_nr = 0
        self.game_nr = 0
        self.in_game_time_total = {bot.name: 0 for bot in bots}
        self.optimize_counter = {bot.name: 0 for bot in bots}

        # create locks for multithreading
        self.data_lock = Lock()
        self.model_lock = Lock()
        self.new_data = False
        self.stop = False

        # initialise important stuff
        self.run_name = run_name
        self.bots: List[MicroBot] = bots
        self.models = {bot.name: bot.model for bot in bots if bot.bot_type == "RL"}
        self.replay_buffer: dict[str, dict] = {bot.name: {"action": [], "location": [], "target": [], "value": [],
                           "observation": [], "lstm_state": [], "reward": [], "done": [], "discounted_reward": [],
                           "old_action_logit": [], "old_location_logit": [], "old_target_logit": []} for bot in bots if bot.bot_type == "RL"}

        self.optimizers = {bot.name: torch.optim.Adam(self.models[bot.name].parameters(), lr=self.settings.learning_rate_start) for bot in bots if bot.bot_type == "RL"}
        self.lr_scheduler = {bot.name: ExponentialLR(self.optimizers[bot.name], gamma=self.settings.learning_rate_decay) for bot in bots if bot.bot_type == "RL"}
        self.eps_clip = {bot.name: self.settings.eps_clip_start for bot in bots if bot.bot_type == "RL"}
        self.state_dicts = {bot.name: bot.model.state_dict() for bot in self.bots if bot.bot_type == "RL"}

        # logging data
        self.output_dirs = {bot.name: f"runs/{datetime.datetime.now().isoformat(sep='_')[5:-10].replace(':', '-')}_{bot.name}_{self.run_name}" for bot in bots if bot.bot_type == "RL"}
        self.writers = {bot.name: SummaryWriter(self.output_dirs[bot.name]) for bot in bots if bot.bot_type == "RL"}
        self.reward_history = {bot.name: [] for bot in bots}
        self.actor_loss = {bot.name: [] for bot in bots if bot.bot_type == "RL"}
        self.critic_loss = {bot.name: [] for bot in bots if bot.bot_type == "RL"}
        self.entropy_loss = {bot.name: [] for bot in bots if bot.bot_type == "RL"}

        self.wins = {bot.name: 0 for bot in bots}
        self.ties = {bot.name: 0 for bot in bots}
        self.losses = {bot.name: 0 for bot in bots}
        self.winrate_window = 50
        self.last_results = {bot.name: [] for bot in bots}
        self.action_distribution = {bot.name: {action: 0 for action in ActionsML} for bot in bots}

        self.log_counter = {bot.name: 0 for bot in bots}
        self.start_time = time.time()

        for bot in [bot for bot in bots if bot.bot_type == "RL"]:
            os.mkdir(self.output_dirs[bot.name] + "/model_checkpoints")
            with open(f"{self.output_dirs[bot.name]}/model_checkpoints/log_info.txt", "a") as f:
                if bot.load_model:
                    f.write(f"Model loaded from: {bot.load_model}\n")

                    if not self.settings.evaluation_mode:
                        self.initialise_continued_run(bot)
                else:
                    f.write(f"No Model loaded\n")

            self.log_run_info(bot)

    def log_run_info(self, bot:MicroBot):
        # log LearningManager settings
        self.settings.log(self.writers[bot.name])

        # log MicroBot settings
        bot.settings.log(self.writers[bot.name])

        # log BattleModel settings
        bot.model.settings.log(self.writers[bot.name])

        self.scenario_settings.log(self.writers[bot.name])
        self.hardware_settings.log(self.writers[bot.name])

        model_params = sum(param.numel() for param in self.models[bot.name].parameters())
        print(f"nr parameters: {model_params}")

        unit_encoder_params = sum(param.numel() for param in self.models[bot.name].unit_encoder.parameters())
        print(f"unit_encoder_params: {unit_encoder_params}")
        spatial_encoder_params = sum(param.numel() for param in self.models[bot.name].spatial_encoder.parameters())
        print(f"spatial_encoder_params: {spatial_encoder_params}")

        core_params = sum(param.numel() for param in self.models[bot.name].core.parameters())
        print(f"core_params: {core_params}")

        action_head_params = sum(param.numel() for param in self.models[bot.name].action_head.parameters())
        print(f"action_head_params: {action_head_params}")
        location_head_params = sum(param.numel() for param in self.models[bot.name].location_head.parameters())
        print(f"location_head_params: {location_head_params}")
        target_head_params = sum(param.numel() for param in self.models[bot.name].target_head.parameters())
        print(f"target_head_params: {target_head_params}")

        critic_params = sum(param.numel() for param in self.models[bot.name].critic.parameters())
        print(f"critic_params: {critic_params}")

        self.writers[bot.name].add_scalar("Info/model_params", model_params)

        self.writers[bot.name].add_scalar("Info/spatial_input_resolution_x", SPATIAL_RESOLUTION_X)
        self.writers[bot.name].add_scalar("Info/spatial_input_resolution_y", SPATIAL_RESOLUTION_Y)
        self.writers[bot.name].add_scalar("Info/spatial_features", SPACIAL_FEATURES)


        self.writers[bot.name].add_scalar("Info/game_nr", self.game_nr, (time.time() - self.start_time)/60)

    def initialise_continued_run(self, bot:MicroBot):
        log_counter = int(re.match('.*?([0-9]+)$', bot.load_model).group(1))
        info_path = bot.load_model[:re.search("log_counter", bot.load_model).start()] + "log_info.txt"

        with open(info_path, "r") as info_file:
            raw_model_log_info = info_file.readlines()
            for line in raw_model_log_info[1:]:
                numbers = line.strip().split(', ')
                log_info = {"log_counter": int(numbers[0]), "optimize_counter": int(numbers[1]), "run_time": float(numbers[2]),
                            "episode_nr": int(numbers[3]), "game_nr": int(numbers[4]), "in_game_time": int(numbers[5][:-2])}
                if log_info["log_counter"] == log_counter:
                    self.log_counter[bot.name] = log_info["log_counter"]
                    self.optimize_counter[bot.name] = log_info["optimize_counter"]
                    self.start_time = time.time() - (log_info["run_time"]*60)
                    self.episode_nr = log_info["episode_nr"]
                    self.game_nr = log_info["game_nr"]
                    self.in_game_time_total[bot.name] = log_info["in_game_time"]*60

    def add_data(self, bot:MicroBot, episode: Episode):
        self.reward_history[bot.name].append(episode.total_reward)

        rewards, dones = episode.trajectory["reward"], episode.trajectory["done"]

        discounted_rewards = []
        discounted_reward = 0

        for i, (reward, done) in enumerate(zip(reversed(rewards), reversed(dones))):
            if done:
                discounted_reward = 0

            discounted_reward = reward + self.settings.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        for key in episode.trajectory.keys():
            self.replay_buffer[bot.name][key] += episode.trajectory[key]
        self.replay_buffer[bot.name]["discounted_reward"] += discounted_rewards

        self.writers[bot.name].add_scalar("Extra_data/all_reward", episode.total_reward, self.episode_nr)


    def received_data(self):
        with self.data_lock:
            self.episode_nr += 1
            self.new_data = True

            for bot in [microbot for microbot in self.bots if microbot.bot_type == "RL"]:
                # remove the oldest data if we reached max memory size
                while len(self.replay_buffer[bot.name]["action"]) > self.settings.memory_size:
                    for key in self.replay_buffer[bot.name].keys():
                        self.replay_buffer[bot.name][key].pop(0)

                # logging stuff
                while len(self.last_results[bot.name]) > self.winrate_window:
                    self.last_results[bot.name].pop(0)

                self.writers[bot.name].add_scalar("Extra_data/wins", self.wins[bot.name], self.episode_nr)
                self.writers[bot.name].add_scalar("Extra_data/ties", self.ties[bot.name], self.episode_nr)
                self.writers[bot.name].add_scalar("Extra_data/losses", self.losses[bot.name], self.episode_nr)
                self.writers[bot.name].add_scalar("Extra_data/win_rate", sum(self.last_results[bot.name]) / self.winrate_window, self.episode_nr)

                for action in ActionsML:
                    self.writers[bot.name].add_scalar(f"Extra_data/{action.name}", self.action_distribution[bot.name][action], self.episode_nr)

                if len(self.reward_history[bot.name]) > self.settings.reward_logging_frequency:
                    average_reward = sum(self.reward_history[bot.name]) / len(self.reward_history[bot.name])
                    print("logging reward:", average_reward)
                    self.writers[bot.name].add_scalar("Reward/reward", average_reward, self.episode_nr)
                    self.writers[bot.name].add_scalar("Reward/reward_time", average_reward, (time.time() - self.start_time)/60)
                    self.reward_history[bot.name] = []

                    self.writers[bot.name].add_scalar("Extra_data/ingame_time", self.in_game_time_total[bot.name] // 60, (time.time() - self.start_time) / 60)

    def increment_game_nr(self):
        self.game_nr += 1
        for bot in [bot for bot in self.bots if bot.bot_type == "RL"]:
            self.writers[bot.name].add_scalar("Info/game_nr", self.game_nr, (time.time() - self.start_time)/60)

    async def training_loop(self):
        while True:
            if self.new_data:
                self.new_data = False
                for bot in [bot for bot in self.bots if bot.bot_type == "RL"]:
                    try:
                        await self.optimize(bot)
                    except Exception as e:
                        print(print(f"{Fore.RED}Error optimizing: {e}{Style.RESET_ALL}"))


            await asyncio.sleep(self.settings.time_between_optimizations)

    def prepare_data(self, bot: MicroBot, batch_size: int):
        idxs = random.sample(range(len(self.replay_buffer[bot.name]["action"])), batch_size)

        actions = np.array([self.replay_buffer[bot.name]["action"][i] for i in idxs])
        locations = np.array([self.replay_buffer[bot.name]["location"][i] for i in idxs])
        targets = np.array([self.replay_buffer[bot.name]["target"][i] for i in idxs])
        values = np.array([self.replay_buffer[bot.name]["value"][i] for i in idxs])
        discounted_rewards = np.array([self.replay_buffer[bot.name]["discounted_reward"][i] for i in idxs])

        observations = [self.replay_buffer[bot.name]["observation"][i] for i in idxs]
        lstm_states = [self.replay_buffer[bot.name]["lstm_state"][i] for i in idxs]

        old_action_logits = np.array([self.replay_buffer[bot.name]["old_action_logit"][i] for i in idxs])
        old_location_logits = np.array([self.replay_buffer[bot.name]["old_location_logit"][i] for i in idxs])
        old_target_logits = np.array([self.replay_buffer[bot.name]["old_target_logit"][i] for i in idxs])

        return actions, locations, targets, values, lstm_states, observations, discounted_rewards, old_action_logits, old_location_logits, old_target_logits

    async def optimize(self, bot: MicroBot):
        if self.settings.evaluation_mode:
            return

        print("optimizer function")
        optimizer = self.optimizers[bot.name]
        optimizer.zero_grad()

        if len(self.replay_buffer[bot.name]["action"]) < self.settings.batch_size:
            print("not enough data yet to optimize")
            return

        total_actor_loss = 0
        total_entropy_loss = 0
        total_critic_loss = 0

        for epoch in range(self.settings.epochs):
            with self.data_lock:
                data = self.prepare_data(bot, self.settings.batch_size)

            actions, locations, targets, old_values, lstm_states, observations, discounted_rewards, old_action_logits, old_location_logits, old_target_logits = data

            discounted_rewards = torch.tensor(discounted_rewards).to(device)
            own_units = torch.stack([obs[0] for obs in observations]).to(device)
            enemy_units = torch.stack([obs[1] for obs in observations]).to(device)
            spatial_features = torch.stack([obs[2] for obs in observations]).to(device)
            scaler_features = torch.stack([obs[3] for obs in observations]).to(device)

            h_0_states = torch.stack([state[0] for state in lstm_states]).to(device).squeeze(1).permute(1, 0, 2)
            c_0_states = torch.stack([state[1] for state in lstm_states]).to(device).squeeze(1).permute(1, 0, 2)

            actions = torch.tensor(actions).unsqueeze(2).to(device)
            locations = torch.tensor(locations).unsqueeze(2).to(device)
            targets = torch.tensor(targets).unsqueeze(2).to(device)
            old_values = torch.tensor(old_values).to(device)
            old_action_logits = torch.tensor(old_action_logits).to(device)
            old_location_logits = torch.tensor(old_location_logits).to(device)
            old_target_logits = torch.tensor(old_target_logits).to(device)
            advantages = discounted_rewards - old_values


            optimizer.zero_grad()
            try:
                action_logits, location_logits, target_logits, critic_values, _ = self.models[bot.name](own_units, enemy_units, spatial_features,
                                                                                          scaler_features, (h_0_states, c_0_states))
            except Exception as e:
                print(print(f"{Fore.RED}{e}{Style.RESET_ALL}"))

            action_probs = action_logits.gather(dim=2, index=actions).squeeze()
            location_probs = location_logits.gather(dim=2, index=locations).squeeze()
            target_probs = target_logits.gather(dim=2, index=targets).squeeze()

            old_action_probs = old_action_logits.gather(dim=2, index=actions).squeeze()
            old_location_probs = old_location_logits.gather(dim=2, index=locations).squeeze()
            old_target_probs = old_target_logits.gather(dim=2, index=targets).squeeze()

            ratios_actions = torch.div(action_probs, old_action_probs).mean(dim=-1)
            ratios_locations = torch.div(location_probs, old_location_probs).mean(dim=-1)
            ratios_targets = torch.div(target_probs, old_target_probs).mean(dim=-1)

            # entropy loss
            entropy_loss_action = -torch.mean(action_logits * torch.log(action_logits), dim=(1, 2))
            entropy_loss_location = -torch.mean(location_logits * torch.log(location_logits), dim=(1, 2))
            entropy_loss_target = -torch.mean(target_logits * torch.log(target_logits), dim=(1, 2))

            # surrogate actor loss
            actions_surr_loss1 = ratios_actions * advantages.squeeze()
            actions_surr_loss2 = torch.clamp(ratios_actions, 1 - self.eps_clip[bot.name], 1 + self.eps_clip[bot.name]) * advantages

            location_surr_loss1 = ratios_locations * advantages.squeeze()
            location_surr_loss2 = torch.clamp(ratios_locations, 1 - self.eps_clip[bot.name], 1 + self.eps_clip[bot.name]) * advantages

            target_surr_loss1 = ratios_targets * advantages.squeeze()
            target_surr_loss2 = torch.clamp(ratios_targets, 1 - self.eps_clip[bot.name], 1 + self.eps_clip[bot.name]) * advantages

            action_loss = -torch.min(actions_surr_loss1, actions_surr_loss2)
            location_loss = -torch.min(location_surr_loss1, location_surr_loss2)
            target_loss = -torch.min(target_surr_loss1, target_surr_loss2)

            # critic loss
            critic_loss = 0.5 * ((discounted_rewards - critic_values.squeeze()) ** 2)

            # take mean over the batch and add togheter for total loss
            actor_loss = (action_loss + location_loss + target_loss).mean(dim=-1)
            critic_loss = critic_loss.mean(dim=-1)
            entropy_loss = (entropy_loss_action + entropy_loss_location + entropy_loss_target).mean(dim=-1) * 0.01
            loss = actor_loss + critic_loss + entropy_loss

            # backpropagation + gradient decent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[bot.name].parameters(), self.settings.gradient_clip)
            optimizer.step()

            total_actor_loss += actor_loss.detach().to('cpu')
            total_entropy_loss += entropy_loss.detach().to('cpu')
            total_critic_loss += critic_loss.detach().to('cpu')

        print("saving state dict")
        self.state_dicts[bot.name] = self.models[bot.name].state_dict()

        # updating lr and eps clip
        if self.lr_scheduler[bot.name].get_last_lr()[0] > self.settings.minimum_learning_rate:
            self.lr_scheduler[bot.name].step()

        self.eps_clip[bot.name] *= self.settings.eps_clip_decay
        if self.eps_clip[bot.name] < self.settings.eps_clip_min:
            self.eps_clip[bot.name] = self.settings.eps_clip_min

        # logging
        self.optimize_counter[bot.name] += 1
        self.actor_loss[bot.name].append(total_actor_loss / self.settings.epochs)
        self.critic_loss[bot.name].append(total_critic_loss / self.settings.epochs)
        self.entropy_loss[bot.name].append(total_entropy_loss / self.settings.epochs)

        # write data every `loss_logging_frequency` training loops
        if len(self.actor_loss[bot.name]) > self.settings.loss_logging_frequency:
            avg_actor_loss = sum(self.actor_loss[bot.name]) / len(self.actor_loss[bot.name])
            avg_critic_loss = sum(self.critic_loss[bot.name]) / len(self.critic_loss[bot.name])
            avg_entropy_loss = sum(self.entropy_loss[bot.name]) / len(self.entropy_loss[bot.name])

            print(f"critic loss: {avg_critic_loss}, actor loss: {avg_actor_loss}")
            self.writers[bot.name].add_scalar("Actor_Loss/train", avg_actor_loss, self.optimize_counter[bot.name])
            self.writers[bot.name].add_scalar("Critic_Loss/train", avg_critic_loss, self.optimize_counter[bot.name])
            self.writers[bot.name].add_scalar("Entropy_Loss/train", avg_entropy_loss, self.optimize_counter[bot.name])

            self.writers[bot.name].add_scalar("Actor_Loss/train_time", avg_actor_loss, (time.time() - self.start_time)//60)
            self.writers[bot.name].add_scalar("Critic_Loss/train_time", avg_critic_loss, (time.time() - self.start_time)//60)
            self.writers[bot.name].add_scalar("Entropy_Loss/train_time", avg_entropy_loss, (time.time() - self.start_time)//60)

            self.writers[bot.name].add_scalar("Extra_data/learning_rate", self.lr_scheduler[bot.name].get_last_lr()[0], self.optimize_counter[bot.name])
            self.writers[bot.name].add_scalar("Extra_data/eps_clip", self.eps_clip[bot.name], self.optimize_counter[bot.name])

            self.actor_loss[bot.name] = []
            self.critic_loss[bot.name] = []

            self.log_counter[bot.name] += 1

            if self.log_counter[bot.name] % self.settings.model_save_frequency == 0:
                torch.save(self.models[bot.name], self.output_dirs[bot.name] + "/model_checkpoints/" + "log_counter" + str(self.log_counter[bot.name]))

                with open(f"{self.output_dirs[bot.name]}/model_checkpoints/log_info.txt", 'a') as f:
                    f.write(f"{self.log_counter[bot.name]}, {self.optimize_counter[bot.name]}, {(time.time() - self.start_time)//60}, {self.episode_nr}, {self.game_nr}, {self.in_game_time_total[bot.name]//60}\n")

            self.writers[bot.name].flush()

