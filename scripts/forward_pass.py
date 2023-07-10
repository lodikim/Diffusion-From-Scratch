import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
main_path = os.path.dirname(dir_path)
sys.path.append(main_path)

import torch
from torch.utils.data import DataLoader
from noise_scheduler import forward_diffusion_sample
from data_preprocessing import load_transformed_dataset, show_tensor_image
import matplotlib.pyplot as plt


# Load and preprocess dataset
BATCH_SIZE = 128
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Simulate forward diffusion
T = 300
image = next(iter(dataloader))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img, 'forward_pass')