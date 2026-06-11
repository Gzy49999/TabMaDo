import pandas as pd
import torch
import os
import numpy as np
from copy import deepcopy
import time

#平滑地更新目标参数，使其逐渐接近源参数
def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average（指数移动平均）.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate（EMA衰减率） (closer to 1 means slower).
    使用以下公式更新目标参数：target_param=target_param×rate+source_param×(1−rate)
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, save_path=None, num_checkpoints=1, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        #创建一个去噪函数的副本，用于 EMA 更新
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        #将 EMA 模型的参数从计算图中分离，避免梯度计算
        for param in self.ema_model.parameters():
            param.detach_()
        #self.n_dim = n_dim    
        self.is_cond = self.diffusion._denoise_fn.is_cond
        if self.is_cond:
            print("Conditional Training!")
        #训练数据的迭代器
        self.train_iter = train_iter
        #总训练步数
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step','loss'])
        #每 100 步记录一次损失值
        self.log_every = 100
        #每 500 步打印一次训练信息
        self.print_every = 500
        #每 1000 步更新一次 EMA 模型
        #self.ema_every = 1000
        #每 steps//num_checkpoints 步保存一次模型检查点
        self.step_per_check = steps//num_checkpoints
        self.save_path = save_path

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "checkpoints/"), exist_ok=True)

    #根据训练进度调整学习率
    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    #执行一次训练步骤
    def _run_step(self, x, cond=None, epsilon=None):
        x = x.to(self.device)
        if self.is_cond and cond is not None:
            cond = cond.to(self.device)
        #清零优化器的梯度
        self.optimizer.zero_grad()

        loss = self.diffusion.compute_loss(x, cond)
        #调用 backward 方法进行反向传播，计算梯度
        loss.backward()
        # for p in self.diffusion.parameters():
        # #     print(p)
        # #     print(p.grad)
        #      p.grad.clamp_(-0.1, 0.1)
        #调用 step 方法更新优化器的参数
        self.optimizer.step()
        return loss

    #执行整个训练循环
    def run_loop(self):
        step = 0
        curr_loss= 0.0

        curr_count = 0
        train_start = time.time()
        while step < self.steps:
            if self.is_cond:
                x, cond = next(self.train_iter)
            else:
                x = next(self.train_iter)[0]
                cond = None

            batch_loss = self._run_step(x, cond)

            #print(f"Step {step}:", (batch_loss_gauss + batch_loss_multi).data)
            self._anneal_lr(step)
            #记录了从上次记录损失以来处理的总样本数量
            curr_count += len(x)
            # batch_loss.item()：获取当前批次的损失值，并将其转换为Python的标量值。
            # len(x)：获取当前批次的样本数量。
            # curr_loss记录了从上次记录损失以来所有批次的总损失值
            curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                #计算平均损失
                loss = np.around(curr_loss/ curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}')
                self.loss_history.loc[len(self.loss_history)] = [step + 1, loss]
                curr_count = 0
                curr_loss = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            if step % self.step_per_check == 0 and self.save_path is not None:
                torch.save(self.diffusion, os.path.join(self.save_path, f"checkpoints/diff_model_{step}.pt"))
        train_end = time.time()

        self.loss_history.loc[len(self.loss_history)] = [step, train_end-train_start]
        if self.save_path is not None:
            self.loss_history.to_csv(os.path.join(self.save_path, "loss_history.csv"), index=None)
