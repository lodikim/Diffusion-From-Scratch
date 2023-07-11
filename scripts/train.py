import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
main_path = os.path.dirname(dir_path)
sys.path.append(main_path)

import torch
from model import model
from torch.optim import Adam
from loss import get_loss
from sampling import sample_plot_image

from torch.utils.data import DataLoader
from data_preprocessing import load_transformed_dataset

# Load and preprocess dataset
BATCH_SIZE = 128
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!
T = 300
BATCH_SIZE = 128

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if epoch % 2 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image(f"epoch_{epoch}")