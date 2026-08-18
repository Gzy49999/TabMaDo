import sys
sys.path.append('../')
import torch


def get_cond_fn(guider, target_cls,train_x_min, guide_strength):
    def cond_fn(x_t,t):
        x_t = x_t.float()
        with torch.enable_grad():
            cond_anchor = train_x_min.unsqueeze(0).repeat(x_t.shape[0], 1)
            gradients = guider.get_guide_gradients(x_t, t, cond_anchor,target_cls,guide_strength)
            return gradients
    return cond_fn

def oversampling(num_samples,diffuser, guider, train_x_min, device,guide_strength=0.00001):
    diffuser.to(device)
    guider.to(device)
    diffuser.variables_to_device(device)
    target_cls = 1
    cond_fn = get_cond_fn(guider,target_cls,train_x_min,guide_strength)
    samples = diffuser.batch_sample(num_samples,cond_fn, batch_size=1000)
    return samples
