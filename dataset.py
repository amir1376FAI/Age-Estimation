import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os


# ******************************** Split datasets ********************************

df_train, df_valid_test = train_test_split(df, test_size=0.3, stratify=df.age , random_state=42)

df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, stratify=df_valid_test.age , random_state=46)

df_train.to_csv('/content/train_set.csv', index=False)
df_valid.to_csv('/content/valid_set.csv', index=False)
df_test.to_csv('/content/test_set.csv', index=False)

print('All CSV files created successfully.')


# ******************************** Define transformations ********************************
transform_train = T.Compose([
                             T.Resize((128, 128)),
                             T.RandomHorizontalFlip(p=0.5),
                             T.RandomRotation(degrees=15),
                             T.ColorJitter(brightness=2, contrast=0.2, saturation = 0.2, hue = 0.1),
                             T.ToTensor(),
                             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                             ])

transform_valid = T.Compose([
                             T.Resize((128, 128)),
                             T.ToTensor(),
                             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                             ])

#  ******************************** Custom dataset ********************************
class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data['image_name'].iloc[idx])

        age = self.data['age'].iloc[idx]
        age = torch.tensor([age], dtype = torch.float32)
        # gender = self.data['gender'].iloc[idx]
        # ethnicity = self.data['ethnicity'].iloc[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)



        return image, age
        # , gender, ethnicity

train_set = UTKDataset('/content/UTKFace', '/content/train_set.csv', transform_train)

valid_set = UTKDataset('/content/UTKFace', '/content/valid_set.csv', transform_valid)

test_set = UTKDataset('/content/UTKFace', '/content/test_set.csv', transform_valid)

# ******************************** Define dataloader ********************************
train_loader = DataLoader(train_set, batch_size = 128, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = 256, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 256, shuffle = False)

# Test the dataloaders
train_iter = iter(train_loader)
X, y = next(train_iter)
X.shape, y.shape
random_idx = torch.randint(len(X), size = (1,))
plt.imshow(X[random_idx.item()].permute(1, 2, 0));


