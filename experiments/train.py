import pytorch_lightning as pl

from task import *
from dataset import *

def main():
    dir = 'C:/Users/scy02/projects/project-animal/experiments/afhq'
    output_dir = 'C:/Users/scy02/projects/project-animal'

    train_csv = f"{dir}/annotations_train.csv"
    test_csv = f"{dir}/annotations_test.csv"
    # train_csv = f"{dir}/annotations_train2.csv"
    # test_csv = f"{dir}/annotations_test2.csv"
    train_dir = f"{dir}/train"
    test_dir = f"{dir}/test"

    es = pl.callbacks.EarlyStopping(monitor = "val_acc", verbose = True, patience = 5, mode = 'max')
    ckpt = pl.callbacks.ModelCheckpoint(dirpath = f"{output_dir}/ckpt/", filename = '{epoch}-{val_acc:.4f}',monitor="val_acc",verbose = True, save_top_k = 3, mode = 'max')

    AnimalBB = AnimalBackBone()
    AnimalDM = AnimalDataModule(train_csv, train_dir, test_csv, test_dir, batch_size = 32, device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trainer = pl.Trainer(callbacks=[ckpt, es], accelerator="gpu" if torch.cuda.is_available else "cpu", val_check_interval=0.5, num_sanity_val_steps=2, max_epochs = -1)
    trainer.fit(AnimalBB,datamodule=AnimalDM)
    trainer.test(AnimalBB, datamodule = AnimalDM)

if __name__=="__main__":
    main()