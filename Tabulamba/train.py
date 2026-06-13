import pandas as pd
import torch
import os
import numpy as np
from copy import deepcopy
import time
import matplotlib.pyplot as plt

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, save_dir,num_checkpoints=1, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'total_loss', 'loss_class_0', 'loss_class_1'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000
        self.step_per_check = steps//num_checkpoints
        self.save_dir = save_dir

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "checkpoints/"), exist_ok=True)


    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, y, cond=None, epsilon=None):
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()

        # loss = self.diffusion.compute_loss(x)
        loss, loss_per_class = self.diffusion.compute_loss_with_class_stats(x, y)
        loss.backward()
        self.optimizer.step()

        return loss, loss_per_class

    def run_loop(self):
        step = 0
        curr_loss= 0.0
        curr_count = 0

        class_loss_sum = {0: 0.0, 1: 0.0}
        class_count = {0: 0, 1: 0}

        train_start = time.time()
        while step < self.steps:
            batch = next(self.train_iter)[0]

            x = batch
            y = batch[:, -1]

            # batch_loss = self._run_step(x)
            batch_loss, loss_per_class = self._run_step(x, y)

            for class_id, loss_val in loss_per_class.items():
                if loss_val is not None:
                    class_loss_sum[class_id] += loss_val * (y == class_id).sum().item()
                    class_count[class_id] += (y == class_id).sum().item()

            curr_count += len(x)
            curr_loss += batch_loss.item() * len(x)

            self._anneal_lr(step)

            if (step + 1) % self.log_every == 0:
                loss = np.around(curr_loss/ curr_count, 4)
                class_avg_loss = {}
                for class_id in class_loss_sum:
                    if class_count[class_id] > 0:
                        class_avg_loss[class_id] = np.around(
                            class_loss_sum[class_id] / class_count[class_id], 4
                        )
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}')
                    print(f'  Class-wise MSE: majority(0)={class_avg_loss.get(0, "N/A")}, '
                          f'minority(1)={class_avg_loss.get(1, "N/A")}, '
                          f'minority(2)={class_avg_loss.get(2, "N/A")}')
                record = {'step': step + 1,
                          'total_loss': loss,
                          'loss_class_0': class_avg_loss.get(0, None),
                          'loss_class_1': class_avg_loss.get(1, None)
                          }
                self.loss_history.loc[len(self.loss_history)] = record

                curr_count = 0
                curr_loss = 0.0
                class_loss_sum = {0: 0.0, 1: 0.0}
                class_count = {0: 0, 1: 0}
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            if step % self.step_per_check == 0 and self.save_dir is not None:
                torch.save(self.diffusion, os.path.join(self.save_dir, f"checkpoints/diff_model_{step}.pt"))
        train_end = time.time()

        final_record = {
            'step': step,
            'total_loss': train_end - train_start,
            'loss_class_0': None,
            'loss_class_1': None
        }
        self.loss_history.loc[len(self.loss_history)] = final_record
        if self.save_dir is not None:
            self.loss_history.to_csv(os.path.join(self.save_dir, "loss_history.csv"), index=None)

