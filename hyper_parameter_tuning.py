from torch import nn
from torch.utils.data import DataLoader, random_split
from torch import optim



from custom_dataset_dataloader import train_set, train_loader
from functions import train_one_epoch
from model import AgeEstimationModel

from prettytable import PrettyTable
import pandas as pd

import torchmetrics as tm

# ************************ Config ************************
loss_fn = nn.L1Loss()
metric = tm.MeanAbsoluteError().to(device)

# ************************ Calculate the loss for an untrained model using a few batches  ************************
                  
model = AgeEstimationModel().to(device)
loss_fn = nn.L1Loss()

optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = 1e-4)

input, target = next(iter(train_loader))
input = input.to(device)
target = target.to(device)


with torch.no_grad():
    output = model(input)
    loss = loss_fn(output, target)
print(loss)

# ************************ Try to train and overfit the model on a small subset of the dataset ************************

_, mini_train_dataset = random_split(train_set, (len(train_set)-1000, 1000))
mini_train_loader = DataLoader(dataset=mini_train_dataset,
                    batch_size=128,
                    shuffle=True)

model = AgeEstimationModel().to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-1, momentum=0.9, weight_decay = 1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    model, loss, metric_mae = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, metric, epoch)


# ************************ Train the model for a limited number of epochs, experimenting with various learning rates ************************

num_epochs = 3
table_best_lr = PrettyTable([ "Learning Rate", "loss"])

for lr in [0.1, 0.01, 0.001, 0.0001]:
  print(f'LR={lr}')
  model = AgeEstimationModel().to(device)
  optimizer = optim.SGD(model.parameters(), lr =lr, momentum=0.9, weight_decay = 1e-4)

  for epoch in range(num_epochs):
    model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch)

    if epoch == num_epochs-1:
      table_best_lr.add_row([lr, f'{loss:.4f}'])
  print()

print(table_best_lr)

# ************************ Create a small grid using the weight decay and the best learning rate ************************

num_epochs = 5
my_table = PrettyTable([ "Learning Rate", "Weight decay", "accuracy", "loss"])

for lr in [0.005 ,0.003, 0.001, 0.0007, 0.005]:
  for wd in [0, 1e-5, 1e-4]:
    print(f'LR={lr}, WD={wd}')
    model = AgeEstimationModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr =lr, momentum=0.9, weight_decay = wd)

    for epoch in range(num_epochs):

      model, loss, accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch) # without augmentation

      if epoch == num_epochs-1:
        my_table.add_row([lr, wd, f'{100.*accuracy:.4f}', f'{loss:.4f}'])
        loss_valid, acc_valid = evaluate(model,
                                     valid_loader,
                                     loss_fn,
                                     metric)
        print(f'Valid: Loss = {loss_valid:.4}, MAE = {acc_valid:.4}')
        print()

    print()

  my_table.add_row([20*'=', 20*'=', 20*'=', 20*'='])

print(my_table)

