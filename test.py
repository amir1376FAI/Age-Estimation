import torch
from torch import nn, optim
import torchmetrics as tm

import os
import matplotlib.pyplot as plt

from config import config
from model import AgeEstimationModel
from custom_dataset_dataloader import test_loader
import face_recognition


# ****************************** Test ******************************
model = torch.load('/content/drive/MyDrive/model_age_estimation.pt', weights_only=False).to(device)
model.eval()

_, metric_test = evaluate(model, test_loader, loss_fn, metric)

print(f'Train: MAE = {metric_train:.4}')
print(f'Valid: MAE = {metric_valid:.4}')
print(f'Test: MAE = {metric_test:.4}')

# ******************************  Inference   ******************************
def inference(image_path, transform, model, face_detection = False):
  if face_detection:
    img = face_recognition.load_image_file(image_path)
    top, right, bottom, left =  face_recognition.face_locations(img)[0]
    img_crop = img[top:bottom, left:right]
    img_crop = Image.fromarray(img_crop)

  else:
    img_crop = Image.open(image_path).convert('RGB')

  img_tesnor = transform(img_crop).unsqueeze(0).to(device)
  model.eval()
  with torch.inference_mode():
    preds = model(img_tesnor).item()
  return preds, img_crop

# ****************************** Load a random from a folder ******************************
folder_image_path = '/content/UTKFace'
image_files = os.listdir(folder_image_path)

rand_idx = random.randint(0, len(image_files))
test_image_path = os.path.join(folder_image_path, image_files[rand_idx])
prediction_age, images = inference(test_image_path, transform_valid, model, False)

real_age = image_files[rand_idx].split('_')[0]
print(f'Real age: {real_age}, Predicted:{prediction_age}')
images

