from dataclasses import dataclass


class Episode:
    def __init__(self):
        self.trajectory = {"action": [], "location": [], "target": [], "value": [],
                           "observation": [], "lstm_state": [], "reward": [], "done": [],
                           "old_action_logit": [], "old_location_logit": [], "old_target_logit": []}
        self.winner = None

    @property
    def total_reward(self):
        return sum(self.trajectory["reward"])

    @property
    def length(self):
        return len(self.trajectory["action"])