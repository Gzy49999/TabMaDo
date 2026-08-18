import sys

from ddpm.modules import MLPClassifierWithPrototype

sys.path.append('../')
from ddpm import modules, diffusion, train
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import data_utils as du
from ddpm.resample import create_named_schedule_sampler
import os

def data_preprocessing(raw_data, label, save_dir=None):
    data_wrapper = du.DataWrapper()
    label_wrapper = du.DataWrapper()
    data_wrapper.fit(raw_data)
    label_wrapper.fit(raw_data[[label]])

    if save_dir is not None:
        du.save_pickle(data=data_wrapper, path=os.path.join(save_dir, 'data_wrapper.pkl'))
        du.save_pickle(data=label_wrapper, path=os.path.join(save_dir, 'label_wrapper.pkl'))
    return data_wrapper, label_wrapper

def set_anneal_lr(opt, init_lr, step, all_steps):
	frac_done = step / all_steps
	lr = init_lr * (1 - frac_done)
	for param_group in opt.param_groups:
		param_group["lr"] = lr

def diffuser_training(train_x, save_dir, device, num_timesteps=1000, epochs=30000, lr=0.0018,  bs=4096):
    train_x = torch.from_numpy(train_x).float()
    model = modules.TabulambaDiffusion(train_x.shape[1], dimension = 1, num_classes=2,)
    model.to(device)
    print("Tabulamba Model Initialization")
    
    diff_model= diffusion.GaussianDiffusion(train_x.shape[1], model, device=device, num_timesteps=num_timesteps)
    diff_model.to(device)
    diff_model.train()
    print("Diffusion Initialization")
    ds = [train_x]
    dl = du.prepare_fast_dataloader(ds, batch_size = bs, shuffle = True)

    trainer = train.Trainer(diff_model, dl, lr, 0.0, epochs, save_dir =save_dir , device=device)
    train_sta = time.time()
    trainer.run_loop()
    train_end = time.time()
    print(f'training time: {train_end-train_sta}')
    
    diff_model.to(torch.device('cuda:0'))
    diff_model.variables_to_device(torch.device('cuda:0'))
    diff_model.eval()
    torch.save(diff_model,(os.path.join(save_dir, 'diffuser.pt')))


def train_guider(
    train_x, train_y, diffuser, save_path, device, n_classes=2, lr=0.001,
    d_hidden=[512, 512], steps=10000, drop_out=0.0, bs=1024,
    lambda_contrast=0.5,
    margin=0.3,
    pos_weight=3.0,
	dynamic_margin=True,
):
	initial_margin = margin
	final_margin = 0.5
	train_x = torch.from_numpy(train_x).float().to(device)
	train_y = torch.from_numpy(train_y).float().to(device)

	mask = (train_y == 1)
	train_x_min = train_x[mask]

	minority_prototype = train_x_min.mean(dim=0)

	ds = [train_x, train_y]
	dl = du.prepare_fast_dataloader(ds, batch_size=bs, shuffle=True)
	dl_iter = iter(dl)

	diffuser.to(device)
	diffuser.variables_to_device(device)

	model = modules.ClassifierWithCondScorer(
		d_in=train_x.shape[1],
		d_hidden=d_hidden,
		n_classes=n_classes,
		cond_dim=train_x.shape[1],
		dropout=drop_out,
		margin=margin,
		pos_weight=pos_weight
	)
	model.train()
	model.to(device)

	opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
	schedule_sampler = create_named_schedule_sampler("uniform", diffuser.num_timesteps)

	sta = time.time()

	for step in range(steps):
		try:
			x, y = next(dl_iter)
		except StopIteration:
			dl_iter = iter(dl)
			x, y = next(dl_iter)

		x = x.to(device)
		y = y.to(device)

		if n_classes == 2:
			y = y.squeeze()

		cond_anchor = minority_prototype.unsqueeze(0).repeat(len(x), 1)  # (B, d_in)

		t, _ = schedule_sampler.sample(len(x), device)
		x_t = diffuser.gaussian_q_sample(x, t)

		logits_cls, losses = model(
			x_t=x_t,
			t=t,
			cond_anchor=cond_anchor,
			y=y
		)

		total_loss = losses['cls_loss'] + lambda_contrast * losses['contrast_loss']

		opt.zero_grad()
		total_loss.backward()
		opt.step()
		set_anneal_lr(opt, lr, step, steps)

		if dynamic_margin and step > 0 and step % 1000 == 0:
			progress = step / steps
			new_margin = initial_margin + (final_margin - initial_margin) * progress
			new_margin = min(final_margin, new_margin)
			model.margin = new_margin

			if step % 1000 == 0:
				print(f"Step {step}: 边际值更新为 {new_margin:.4f}")

		if (step + 1) % 100 == 0 or step == 0:
			if n_classes == 2:
				pred_cls = (torch.sigmoid(logits_cls) > 0.5).float()

			else:
				pred_cls = logits_cls.argmax(dim=1)

			print(
				f"Step {step + 1}/{steps} | "
				f"Total Loss: {total_loss:.4f} | "
				f"Cls Loss: {losses['cls_loss']:.4f} | "
				f"Proto Loss: {losses['contrast_loss']:.4f} | "
			)

	end = time.time()
	model.eval()
	model.to(torch.device('cpu'))
	torch.save(model, save_path)