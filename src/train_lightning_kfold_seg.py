import argparse
import yaml
from addict import Dict
import wandb
import os
from sklearn.model_selection import StratifiedKFold
import torch
import pandas as pd

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import cv2
import lightning as L
from model_lightning_seg import MyModel
from data_datamodule_seg import WSIDataModule
import random
import numpy as np

def train(dataset_train, dataset_test,conf,k_fold_value):
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    data_dir = conf.dataset.data_dir
    train_file = dataset_train
    dev_file = dataset_test
    test_file = dataset_test
    cache_data = conf.dataset.cache_data
    rescale_factor = conf.dataset.rescale_factor

    num = dev_file.split('.')[0].split('_')[-1]
    tb_exp_name = f'kfold_{k_fold_value}_{conf.dataset.experiment}_{num}'

    # Setting a random seed for reproducibility
    if conf.train_par.random_seed == 'default':
        random_seed = 2024
    else:
        random_seed = conf.train_par.random_seed
        
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create a DataModule
    data_module = WSIDataModule(batch_size=conf.train_par.batch_size, workers=conf.train_par.workers, train_file=train_file, 
                                dev_file=dev_file,test_file=None, data_dir=data_dir, cache_data=cache_data)
    #Image and Label batch: torch.Size([1, 1, 128, 128, 128])
    
    data_module.prepare_data()
    data_module.setup(stage="fit")
    results_path = os.path.join(conf.train_par.results_path, conf.dataset.experiment)
    os.makedirs(results_path, exist_ok=True)
    conf.train_par.results_model_filename = os.path.join(results_path, f'{tb_exp_name}')
    #wandb logger
    wandb_logger = WandbLogger(project="3d_a_unet_seg", entity="giancarlo-guarnizo",config=conf, name=tb_exp_name)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=conf.train_par.patience, verbose=True, mode="min")
    model_checkpoint = ModelCheckpoint(
        filename=conf.train_par.results_model_filename, monitor="val_loss", mode="min"
    )
    lightning_model =MyModel(model_opts=conf.model_opts, train_par=conf.train_par)

    trainer = L.Trainer(
        max_epochs=conf.train_par.epochs, accelerator="auto", devices="auto",logger=wandb_logger,callbacks=[early_stop_callback,model_checkpoint],        
        default_root_dir=results_path
    )
    trainer.fit(model=lightning_model, datamodule=data_module)
    return trainer

if __name__ == "__main__":
    
    trainparser = argparse.ArgumentParser(description='[StratifIAD] Parameters for training', allow_abbrev=False)
    trainparser.add_argument('-c','--config-file', type=str, default='./default_config_train_seg.yaml')
    args = trainparser.parse_args()
    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    k_fold_value=10
    dir_dataset='./data_csv_new' #Contains ten .csv train k-fold files and ten .csv test k-fold files

    elementos = os.listdir(dir_dataset)
    archivos = [os.path.join(dir_dataset, elemento) for elemento in elementos]
    archivos_ordenados = sorted(archivos, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for iteracion in range(0,k_fold_value*2,2):
        print(archivos_ordenados[iteracion])
        print(archivos_ordenados[iteracion+1])

    print("START")
    for iteracion in range(0,k_fold_value*2,2):
        dataset_test=archivos_ordenados[iteracion]
        dataset_train=archivos_ordenados[iteracion+1]
        print(archivos_ordenados[iteracion])
        print(archivos_ordenados[iteracion+1])
        trainer=train(dataset_train, dataset_test,conf,k_fold_value)
        wandb.finish()
