# Lightning DataModule to handle data loading
from networkx import triangles
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import glob
import pickle as pk
import pytorch_lightning as pl



class customDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, cpus, dir,subdirs, change_interval=10):
        super().__init__()
        self.batch_size = batch_size
        self.cpus = cpus
        self.dir = dir
        self.subdirs = subdirs
        self.change_interval = change_interval
        self.epoch_count = 0
        self.prepare_data()

    def prepare_data(self):
        TRAIN_DATA_FILES=glob.glob(self.dir+self.subdirs[0] +"*_train.pt")
        VAL_DATA_FILES=glob.glob(self.dir+self.subdirs[0]+"*_val.pt")
        self.train_dataset = None
        for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
            with open(TRAIN_DATA_FILE, 'rb') as f:
                if self.train_dataset is None:
                    self.train_dataset = pk.load(f)
                else:
                    self.train_dataset+=pk.load(f)

        self.val_dataset = None
        for VAL_DATA_FILE in VAL_DATA_FILES:
            with open(VAL_DATA_FILE, 'rb') as f:
                if self.val_dataset is None:
                    self.val_dataset = pk.load(f)
                else:
                    self.val_dataset+=pk.load(f)   

        print('###############################')            
        print('Training Samples', len(self.train_dataset))
        print('Validation Samples', len(self.val_dataset))
        print('###############################')
        self.train_dataloader()

    def train_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpus)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus)
        #return train_loader, val_loader

    def on_epoch_end(self):
        self.epoch_count += 1
        item=1
        if self.epoch_count % self.change_interval == 0:
            TRAIN_DATA_FILES=glob.glob(self.dir+self.subdirs[item] +"*_train.pt")
            VAL_DATA_FILES=glob.glob(self.dir+self.subdirs[item]+"*_val.pt")
            self.train_dataset = None
            for TRAIN_DATA_FILE in TRAIN_DATA_FILES:
                with open(TRAIN_DATA_FILE, 'rb') as f:
                    if self.train_dataset is None:
                        self.train_dataset = pk.load(f)
                    else:
                        self.train_dataset+=pk.load(f)

            self.val_dataset = None
            for VAL_DATA_FILE in VAL_DATA_FILES:
                with open(VAL_DATA_FILE, 'rb') as f:
                    if self.val_dataset is None:
                        self.val_dataset = pk.load(f)
                    else:
                        self.val_dataset+=pk.load(f)   

            print('###############################')            
            print('Training Samples', len(self.train_dataset))
            print('Validation Samples', len(self.val_dataset))
            print('###############################')
            item+=1
            self.train_dataloader()  # Update DataLoader