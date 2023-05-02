import torch
import pandas as pd
import pytorch_lightning as pl
import PIL
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, df, dir, device = torch.device('cpu'), transform = None):
        super().__init__()
        self.df = df
        self.dir = dir
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"{self.dir}/({self.df.image_id[idx]}).jpg"
        image = PIL.Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        classes = ["cat", "cheetah", "dog", "fox", "leopard", "lion", "tiger", "wolf"]
        # classes = ["cat", "dog", "fox"]
        return image.to(self.device), torch.tensor([classes.index(self.df["label"][idx])], dtype = torch.long).to(self.device)

class AnimalDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, train_dir, test_csv, test_dir, batch_size, num_workers = 0, train_val_split = 0.8, device = torch.device("cpu")):
        super().__init__()
        self.train_dir = train_dir
        self.train_csv = pd.read_csv(train_csv)
        self.test_dir = test_dir

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        image_ids = self.train_csv["image_id"]
        labels = self.train_csv["label"]

        X_train, X_val, y_train, y_val = train_test_split(image_ids, labels, test_size = 1 - train_val_split)
        self.train_df = pd.DataFrame({"image_id":X_train, "label":y_train}).reset_index(drop = True)
        self.val_df = pd.DataFrame({"image_id":X_val, "label":y_val}).reset_index(drop=True)
        self.test_df = pd.read_csv(test_csv)
        
        image_size = 256
        # image_crop = 312

        self.transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                # transforms.CenterCrop(image_size),
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(AnimalDataset(self.train_df, self.train_dir, self.device, transform = self.transform['train']), shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(AnimalDataset(self.val_df, self.train_dir, self.device, transform = self.transform['val']), shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(AnimalDataset(self.test_df, self.test_dir, self.device, transform = self.transform['val']), shuffle = False, batch_size = self.batch_size, num_workers = self.num_workers)

# dir = 'C:/Users/scy02/projects/project-animal/experiments/afhq'
# output_dir = 'C:/Users/scy02/projects/project-animal'

# train_csv = f"{dir}/annotations_train.csv"
# test_csv = f"{dir}/annotations_test.csv"
# train_dir = f"{dir}/train"
# test_dir = f"{dir}/test"

# dm = AnimalDataModule(train_csv, train_dir, test_csv, test_dir, 4)
# dl = dm.train_dataloader()

# train_features, train_labels = next(iter(dl))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img.numpy().transpose(1,2,0))
# plt.show()
# print(f"Label: {label}")