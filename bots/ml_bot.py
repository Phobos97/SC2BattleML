from sc2.data import Result

from SC2BattleML.bots.micro_bot import MicroBot
from SC2BattleML.models.MicroStar import MicroStar
from SC2BattleML.settings import ModelSettings, MicroBotSettings
import torch
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLBot(MicroBot):
    def __init__(self, name_postfix, settings: MicroBotSettings, model_settings: ModelSettings, observer=False):
        super().__init__(name_postfix=name_postfix, settings=settings, observer=observer)
        self.name = "ML_BOT" + name_postfix
        self.bot_type = "RL"
        self.lstm_state = None

        self.model = MicroStar(model_settings=model_settings, bot_settings=settings).to(device)

        self.load_model = model_settings.load_model
        if self.load_model:
            state_dict = torch.load(self.load_model).state_dict()
            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(print(f"{Fore.RED} Error loading state dict, possible setting mismatch: {e}{Style.RESET_ALL}"))
            else:
                print(f"{Fore.GREEN}LOADED MODEL {self.load_model}{Style.RESET_ALL}")


    async def execute(self):
        observation = await self.get_observation()
        own_units, enemy_units, spatial_features, scalar_features, tags = observation

        prev_lstm_state = self.lstm_state
        if self.lstm_state:
            h_0, c_0 = self.lstm_state
            self.lstm_state = (h_0.to(device), c_0.to(device))

        # get action distributions from neural network
        action_logits, location_logits, target_logits, value, self.lstm_state = self.model(own_units.unsqueeze(0).to(device), enemy_units.unsqueeze(0).to(device),
                                                                            spatial_features.unsqueeze(0).to(device),
                                                                            scalar_features.unsqueeze(0).to(device),
                                                                            self.lstm_state)


        h_0, c_0 = self.lstm_state
        self.lstm_state = (h_0.detach(), c_0.detach())

        if prev_lstm_state:
            h_0, c_0 = prev_lstm_state
            prev_lstm_state = (h_0.detach().to('cpu'), c_0.detach().to('cpu'))
        else:
            prev_lstm_state = (torch.zeros((1, 1, self.model.settings.core_output_size)), torch.zeros((1, 1, self.model.settings.core_output_size)))

        # mask out actions defined in the MicroBotSetting
        action_logits = self.mask_actions(action_logits)

        # sample distributions for final chosen actions
        actions = torch.multinomial(action_logits.squeeze(), num_samples=1)
        locations = torch.multinomial(location_logits.squeeze(), num_samples=1)
        targets = torch.multinomial(target_logits.squeeze(), num_samples=1)

        actions = actions.detach().cpu().numpy().squeeze()
        locations = locations.detach().cpu().numpy().squeeze()
        targets = targets.detach().cpu().numpy().squeeze()

        # apply actions to environment
        await self.execute_actions(actions, locations, targets, tags)

        return (actions, locations, targets), observation, float(value.detach().to('cpu')), prev_lstm_state,\
            (action_logits.squeeze().detach().to('cpu').numpy(), location_logits.squeeze().detach().to('cpu').numpy(), target_logits.squeeze().detach().to('cpu').numpy())

    async def reset_bot(self):
        self.lstm_state = (torch.zeros((1, 1, self.model.settings.core_output_size)), torch.zeros((1, 1, self.model.settings.core_output_size)))

    def mask_actions(self, action_logits):
        for action_type in self.settings.mask_actions:
            action_logits[:, :, action_type.value] = 0
        return action_logits

    async def on_step(self, iteration: int):
        await self.data_collector.on_step(iteration, self)

    async def on_end(self, game_result: Result):
        await self.data_collector.on_end(game_result, self)









