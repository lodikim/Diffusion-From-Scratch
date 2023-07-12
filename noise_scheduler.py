# noise scheduler: sequentially adds noise
# neural network: model that predicts the noise in an image (Unet)
# timestep encoding: a way to encode the current timestep

# Forward Process: add noise to images
# beta: variance schedule (how much noise we want to add) -> can try out different values to check convergence toward a mean of 0
# this code: add noise linearly (quadratic, cosine, sigmoidal, etc. possible)

import torch
import torch.nn.functional as F

# linspace: interpolates between two values -> linear scheduling
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# extracts specific index from a list, considers batch size
def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# main function: calculates the noisy version of an image at a time step t
def forward_diffusion_sample(x_0, t, betas, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    noise = torch.randn_like(x_0)   # sample noise
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)