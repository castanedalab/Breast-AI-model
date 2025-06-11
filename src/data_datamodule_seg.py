from dataset_seg import WSIDataset
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from transforms_seg import ToTensor, Rescale

class WSIDataModule(L.LightningDataModule):
    def __init__(self,batch_size=1, workers=1,train_file=None, dev_file=None, test_file=None,data_dir=None, cache_data=False):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.cache_data = cache_data

    def prepare_data(self):
        pass
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = WSIDataset(meta_data=self.train_file,
                                root_dir=self.data_dir, 
                                cache_data=self.cache_data,
                                transform=transforms.Compose([
                                    ToTensor(),
                                ]))
            self.dev_dataset = WSIDataset(meta_data=self.dev_file,
                                root_dir=self.data_dir,
                                cache_data=self.cache_data,
                                transform=transforms.Compose([ToTensor()]))

        if stage == "test":
            self.test_dataset = WSIDataset(meta_data=self.test_file,
                                    root_dir=self.data_dir,
                                    cache_data=self.cache_data,
                                    transform=transforms.Compose([ToTensor()]))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                            shuffle=False, num_workers=self.workers, pin_memory=True)
