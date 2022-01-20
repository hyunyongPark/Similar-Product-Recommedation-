import numpy as np 
import pandas as pd 

import os 
import cv2 
import timm 

import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
import torch.nn.functional as F 
from torch import nn 
from torch.optim import Adam

import math
import neptune
from sklearn.model_selection import StratifiedKFold

from tqdm.notebook import tqdm 
from sklearn.preprocessing import LabelEncoder

from torch.optim.lr_scheduler import StepLR, ExponentialLR, OneCycleLR, _LRScheduler, ReduceLROnPlateau

from configuration import *
from transform import *
from activation import *
from model import *
from utils import *
from dataset import *

def train_fn(model, data_loader, optimizer, scheduler, i):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for t,data in enumerate(tk):
        for k,v in data.items():
            data[k] = v.to(Config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step() 
        fin_loss += loss.item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})
        
        neptune.log_metric('loss_tr',float(fin_loss/(t+1)))
        neptune.log_metric('train_lr',optimizer.param_groups[0]['lr'])
        
    scheduler.step()

    return fin_loss / len(data_loader)



def eval_fn(model, data_loader, i):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(i+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(Config.DEVICE)
            _, loss = model(**data)
            fin_loss += loss.item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
            
            neptune.log_metric('loss_valid', float(fin_loss/(t+1)))
            
        return fin_loss / len(data_loader)
    
    
    
def run_training():
    seed_setting(Config.SEED)
    neptune.init(project_qualified_name = "younghoon/Deep-rec",
              api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMTAxYTc0NS1kMWFlLTQwNjEtYWQ2OS04ODM3ZGI1YTA2ZjUifQ==",
              )
    #pass parameters to create experiment
    neptune.create_experiment(params=  None , name= Config.MODEL_NAME, description = f'train {Config.EPOCHS}'
                              , tags=['efficientnet_b4','30epochs','one-cycle-lr'] )
    
    
    seed_setting()
    
    df = pd.read_csv(Config.TRAIN_CSV, index_col=0)
    df = df.reset_index(drop=True)
    
    train,valid = stratify_df(df)
    
    #train
    train_dataset = KfashionDataset(train, transform = get_train_transforms())    
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = Config.BATCH_SIZE,
        pin_memory = True,
        num_workers = Config.NUM_WORKERS,
        shuffle = True,
        drop_last = True
    )
    
    #valid 추가
    valid_dataset = KfashionDataset(valid, transform = get_valid_transforms())
    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = Config.BATCH_SIZE,
        num_workers = Config.NUM_WORKERS,
        shuffle = False,
        pin_memory = True,
        drop_last = False
        )
    
    
    model = KfashionModel()
    model.to(Config.DEVICE)
    
    existing_layer = torch.nn.SiLU
    new_layer = Mish()
    model = replace_activations(model, existing_layer, new_layer) # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()
    
    optimizer = Adam(model.parameters(), lr = Config.LR_START)
    scheduler = OneCycleLR(optimizer, max_lr = 2e-3, steps_per_epoch = len(trainloader),epochs=Config.EPOCHS)
    
    save_dir = f'./{Config.SAVE_NAME}_{Config.MODEL_NAME}_{Config.EPOCHS}_Weights'
    
    if os.path.exists(save_dir) == False :
        print('Making Weights Folder')
        os.mkdir(save_dir)
        
    for i in range(Config.EPOCHS):
        avg_loss_train = train_fn(model, trainloader, optimizer, scheduler, i)
        avg_loss_valid = eval_fn(model, validloader,i)
        torch.save(model.state_dict(),f'{save_dir}/{Config.MODEL_NAME}_{i}EpochStep_adamw.pt')
        
if __name__ == '__main__':
    run_training()