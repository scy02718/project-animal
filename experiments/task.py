import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
import numpy as np
import pytorch_lightning as pl

class AnimalBackBone(pl.LightningModule):
    def __init__(self):
        super().__init__()
        num_classes = 8
        # num_classes = 3
        self.weight_decay = 1e-7
        self.model = resnet34(weights = ResNet34_Weights.DEFAULT)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        labels = labels.squeeze()
        features = self(x)
        loss = F.cross_entropy(features, labels)
        self.log("train_loss", loss, on_step = True, on_epoch=True, prog_bar = True, logger = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        labels = labels.squeeze()
        features = self(x)
        prediction = torch.argmax(features, axis = 1)
        val_acc = torch.sum(prediction == labels) / labels.shape[0]
        self.log("val_acc", val_acc)
        return val_acc

    def validation_epoch_end(self, outputs): 
        batch_accs =  [x for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()   
        return {'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_acc: {:.4f}".format( epoch, result['val_acc']))

    def test_step(self, batch, batch_idx):
        x, labels = batch
        labels = labels.squeeze()
        features = self(x)
        prediction = torch.argmax(features, axis = 1)
        test_acc = torch.sum(prediction == labels) / labels.shape[0]
        print(test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=32, verbose = True),
                "monitor":"train_loss",
                "interval":"step"
            }
        }
