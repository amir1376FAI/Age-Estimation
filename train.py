import torch
from torch import nn, optim
import torchmetrics as tm

import os
import matplotlib.pyplot as plt

from config import config
from model import AgeEstimationModel
from functions import train_one_epoch, validation
from custom_dataset_dataloader import train_loader, valid_loader

# ************************** Config **************************
model = AgeEstimationModel().to(device)
lr = 3e-3
wd = 1e-4
optimizer = optim.SGD(model.parameters(), lr =lr, momentum=0.9, weight_decay = wd)
loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

num_epochs = 100

patience = 5
epochs_without_improvement = 0

# ************************** Training Loop **************************
for epoch in range(num_epochs):
  # Train
  model, loss_train, acc_train = train_one_epoch(model,
                                                 train_loader,
                                                 loss_fn,
                                                 optimizer,
                                                 metric,
                                                 epoch)
  # Validation
  loss_valid, acc_valid = evaluate(model,
                                     valid_loader,
                                     loss_fn,
                                     metric)


  loss_train_hist.append(loss_train)
  loss_valid_hist.append(loss_valid)

  acc_train_hist.append(acc_train)
  acc_valid_hist.append(acc_valid)


  if loss_valid < best_loss_valid:
    torch.save(model, f'model.pt')
    best_loss_valid = loss_valid
    print('Model Saved!')
    epochs_without_improvement = 0
  else:
      epochs_without_improvement += 1

  print(f'Valid: Loss = {loss_valid:.4}, MAE = {acc_valid:.4}')
  print()

  if epochs_without_improvement >= patience:
      print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation loss for {patience} epochs.")
      break

  epoch_counter += 1
