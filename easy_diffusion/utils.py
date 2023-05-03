import matplotlib.pyplot as plt
from easy_diffusion.sample import Euler_Maruyama_sampler
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from easy_diffusion.base_model import marginal_prob_std_fn,diffusion_coeff_fn
import numpy as np
import os
device = torch.device('cuda')
import math

def save_samples_cond(score_model, model_name, suffix):
    score_model.eval()
    for digit in range(10):
        ## Generate samples using the specified sampler.
        sample_batch_size = 64 #@param {'type':'integer'}
        num_steps = 500 #@param {'type':'integer'}
        sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
        # score_model.eval()
        ## Generate samples using the specified sampler.
        samples = sampler(score_model, 
            marginal_prob_std_fn,
            diffusion_coeff_fn, 
            sample_batch_size, 
            num_steps=num_steps,
            device=device,
            y=digit*torch.ones(sample_batch_size, dtype=torch.long))

        ## Sample visualization.
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        sample_np = sample_grid.permute(1, 2, 0).cpu().numpy()
        if not os.path.exists(os.path.join("img",model_name)):
            os.makedirs(os.path.join("img", model_name))
        image_file = os.path.join("img", model_name, f"condition_diffusion{suffix}_digit%d.png"%digit)
        plt.imsave(image_file, sample_np,)
        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_np, vmin=0., vmax=1.)
        plt.show()


def save_samples_uncond(score_model, model_name, suffix=""):
    score_model.eval()
    ## Generate samples using the specified sampler.
    sample_batch_size = 64 #@param {'type':'integer'}
    num_steps = 250 #@param {'type':'integer'}
    sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
    # score_model.eval()
    ## Generate samples using the specified sampler.
    samples = sampler(score_model, 
              marginal_prob_std_fn,
              diffusion_coeff_fn, 
              sample_batch_size, 
              num_steps=num_steps,
              device=device,
              )

          ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    sample_np = sample_grid.permute(1, 2, 0).cpu().numpy()
    if not os.path.exists(os.path.join("img", model_name)):
            os.makedirs(os.path.join("img", model_name))
    image_file = os.path.join("img", model_name, f"uncondition_diffusion{suffix}.png")
    plt.imsave(image_file, sample_np,)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_np, vmin=0., vmax=1.)
    plt.show()
    

def visualize_digit_embedding(digit_embed):
    cossim_mat = []
    for i in range(10):
        cossim = torch.cosine_similarity(digit_embed, digit_embed[i:i+1,:]).cpu()
        cossim_mat.append(cossim)
    cossim_mat = torch.stack(cossim_mat)
    cossim_mat_nodiag = cossim_mat + torch.diag_embed(math.nan * torch.ones(10))
    plt.imshow(cossim_mat_nodiag)
    plt.show()
    return cossim_mat