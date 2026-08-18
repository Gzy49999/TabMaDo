import pandas as pd
import lib_oversampling as lo
import data_utils as du
import os
import torch
import argparse
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='diffuser_training')
parser.add_argument('--dataset-name', type=str, default='default')

parser.add_argument('--diffuser-dim', nargs='+', type=int, default=(512, 1024, 1024, 512))
parser.add_argument('--diffuser-lr', type=float, default=0.0018)
# parser.add_argument('--diffuser-lr', type=float, default=0.001)
parser.add_argument('--diffuser-steps', type=int, default=3000)
parser.add_argument('--diffuser-bs', type=int, default=256)
parser.add_argument('--diffuser-timesteps', type=int, default=1000)

parser.add_argument('--controller-dim', nargs='+', type=int, default=(512, 512))
parser.add_argument('--controller-lr', type=float, default=0.0005)
parser.add_argument('--controller-steps', type=int, default=10000)
parser.add_argument('--controller-bs', type=int, default=1325)

parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--scale-factor', type=float, default=8.0)
parser.add_argument('--save-name', type=str, default='output')
args = parser.parse_args()


device = torch.device(f'cuda:{args.device}')

config = du.load_json(f'datasets/dataset_info.json')[args.dataset_name]
label = config['label']
for a in range(1,11):
    diffuser_save_dir = os.path.join('TabMaDo', f"{args.save_name}/{a}")
    guider_save_dir = os.path.join('TabMaDo', f"{args.save_name}/{a}")
    data_save_dir = os.path.join('Ablation_only_label',f"{args.dataset_name}_{a}")
    os.makedirs(diffuser_save_dir, exist_ok=True)
    os.makedirs(guider_save_dir, exist_ok=True)
    os.makedirs(data_save_dir, exist_ok=True)
    train_data = pd.read_csv(f'datasets/raw-data/{args.dataset_name}/{a}-fold/train_data.csv')
    test_data = pd.read_csv(f'datasets/raw-data/{args.dataset_name}/{a}-fold/test_data.csv')
        
    all_data = pd.concat((train_data, test_data))
    data_wrapper, label_wrapper = lo.data_preprocessing(all_data, config['label'], data_save_dir)

    n_classes = len(pd.unique(train_data[label]))

    if args.task_name == 'diffuser_training':
        train_x = data_wrapper.transform(train_data)
        lo.diffuser_training(train_x=train_x,
                             save_dir=diffuser_save_dir,
                             device=device,
                             num_timesteps=args.diffuser_timesteps,
                             epochs=args.diffuser_steps,
                             lr=args.diffuser_lr,
                             bs=args.diffuser_bs)

 

    if args.task_name == 'guider_training':
        diffuser = torch.load(os.path.join(diffuser_save_dir, 'diffuser.pt'))
        train_x = data_wrapper.transform(train_data)
        train_y = label_wrapper.transform(train_data[[label]])

        lo.train_guider(train_x=train_x,
                    train_y=train_y,
                    diffuser=diffuser,
                    save_path=os.path.join(guider_save_dir, 'guider.pt'),
                    device=device,
                    n_classes=n_classes,
                    lr=args.controller_lr,
                    d_hidden=args.controller_dim,
                    steps=args.controller_steps,
                    drop_out=0.0,
                    bs=args.controller_bs,
                    lambda_contrast=0.5,
                    margin=0.5,
                    pos_weight=3.0,
                    dynamic_margin=True
                    )

    if args.task_name == 'oversampling':
        train_x = data_wrapper.transform(train_data)
        diffuser = torch.load(os.path.join(diffuser_save_dir, 'diffuser.pt'))
        guider = torch.load(os.path.join(guider_save_dir, 'guider.pt'))
        mask = train_x[:, -1] == 1
        train_x_min = train_x[mask]
        minority_prototype = torch.tensor(train_x_min.mean(axis=0))
        sample_data = []
        for i in range(len(config['minority_classes'])):
            samples = lo.oversampling(config['n_samples'][i], diffuser, guider, train_x_min, device, args.scale_factor)
            sample_data.append(samples)
        for i in range(len(config['minority_classes'])):
            samples = lo.oversampling(config['n_samples'][i], diffuser, guider,
                                      config['minority_classes'][i], device, args.scale_factor)
            sample_data.append(samples)
        

        raw_sample_data = torch.cat(sample_data, dim=0)
        raw_sample_data = raw_sample_data.cpu().numpy()
        sample_data = data_wrapper.Reverse(raw_sample_data)
        sample_data.to_csv(os.path.join(data_save_dir, 'oversample_data.csv'), index=None)



















