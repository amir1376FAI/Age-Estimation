import torch
from utils import AverageMeter
from tqdm import tqdm

def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=None):
    model.train()
    loss_train = AverageMeter()
    metric.reset()
    with tqdm.tqdm(train_loader, unit="batch", colour='green') as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device) # Move inputs to the device
            targets = targets.to(device) # Move targets to the device



            outputs = model(inputs) # Get outputs from the model

            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n = len(targets))
            metric(outputs, targets.int())
            tepoch.set_postfix(loss=loss_train.avg,
                            MAE=metric.compute().item())
    return model, loss_train.avg, metric.compute().item()



def evaluate(model, test_loader, loss_fn, metric):
    model.eval()
    with torch.inference_mode():
        loss_valid = AverageMeter()
        metric.reset()
        for inputs, targets in test_loader:
            inputs = inputs.to(device) # Move inputs to the device
            targets = targets.to(device) # Move targets to the device

            outputs = model(inputs) # Changed 'input' to 'inputs'
            loss = loss_fn(outputs, targets)

            loss_valid.update(loss.item())
            metric(outputs, targets.int())

    return  loss_valid.avg, metric.compute().item()

