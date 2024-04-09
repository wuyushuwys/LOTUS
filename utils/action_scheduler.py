import time
import numpy as np

units = dict(
    step=1,
    sec=1,
    min=60,
    hour=60 * 60,
    day=60 * 60 * 24,
)

class ActionScheduler:

    def __init__(self, milestones, actions, unit='min') -> None:
        self.milestones = milestones
        self.actions = actions
        assert unit in units.keys()
        assert len(self.milestones) == len(self.actions) - 1
        if unit == 'step':
            self.elapsed_time = 0
        else:
            self.timer = time.monotonic()
        self.unit_name = unit
        self.unit = units[unit]

    def step(self):
        if self.unit_name == 'step':
            elapsed_time = self.elapsed_time
            self.elapsed_time += 1
        else:
            elapsed_time = (time.monotonic() - self.timer) / self.unit # convert based on unit
        return self.actions[np.searchsorted(self.milestones, elapsed_time, side='left')]