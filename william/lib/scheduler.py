import torch


class CustomScheduler:
    def __init__(self, optimizer, schedule: dict = {0: 1e-3}):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self.optimizer = optimizer
        self.intervals = list(schedule.keys())
        self.lrs = list(schedule.values())
        self.num_epoch = 0

    def step(self):
        self._adjust_lr()
        self.num_epoch += 1

    def _adjust_lr(self):
        for idx in range(len(self.intervals)):
            if idx + 1 != len(self.intervals):
                if self.num_epoch >= int(self.intervals[idx]) and self.num_epoch < int(
                    self.intervals[idx + 1]
                ):
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.lrs[idx]
                    break
            else:
                break
