import torch
from noise_scheduler import forward_diffusion_sample
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)   # sampled noise
    noise_pred = model(x_noisy, t)                              # predicted noise
    return F.l1_loss(noise, noise_pred)