import numpy as np 
import pandas as pd 

import math
import random 
import os 
import cv2
import timm
import time

from tqdm import tqdm 

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
from torch.utils.data import Dataset 
from torch import nn
import torch.nn.functional as F 

import matplotlib.pyplot as plt
import plotly.express as px
from urllib.request import urlopen
from PIL import Image
import io
import gc
import requests
from numpy import dot
from numpy.linalg import norm
import json
import pymysql as pymysql

from activation import *
from configuration import *
from model import *
from transform import *
from utils import *



def get_image_embeddings(img_url, model_name = Config.MODEL_NAME):
    embeds = []
    
    model = KfashionModel(model_name = model_name)
    model.eval()
    
    if model_name == 'eca_nfnet_l0':
        model = replace_activations(model, torch.nn.SiLU, Mish())

    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model = model.to(Config.DEVICE)
    
    
    #dt = urlopen(img_url)
    #usr_img = Image.open(io.BytesIO(dt.read())).convert("RGB")
    
    image_nparray = np.asarray(bytearray(requests.get(img_url).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    
    #image = cv2.imread(dt)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    aug = get_test_transforms()
    augmented = aug(image=image)
    img = augmented['image']
    img = img.unsqueeze(0).float()  # if torch tensor
    
    with torch.no_grad():
        img = img.cuda()
        feat = model(img)
        image_embeddings = feat.detach().cpu().numpy()
        embeds.append(image_embeddings)
    
    
    del model
    image_embeddings = np.concatenate(embeds)
    del embeds
    gc.collect()
    return image_embeddings

def get_image_predictions(user_embeddings, train_embs,image_name_emb, threshold):
    
    start = time.time()
    
    # https://gritmind.blog/2020/06/21/diff_similarity_distance/
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    # https://goofcode.github.io/similarity-measure
    # https://www.delftstack.com/ko/howto/python/cosine-similarity-between-lists-python/
    preds = {i: 1- (dot(user_embeddings, vec_b)/(norm(user_embeddings)*norm(vec_b)))[0]
             for i, vec_b in tqdm(enumerate(train_embs))}
    #preds2 = {i : distance.cosine(user_embeddings, vec_b) for i, vec_b in tqdm(enumerate(train_embs))}
    #print(preds2)
    
    preds = dict(sorted(preds.items(), key = lambda item: item[1]
                        #, reverse = True
                       )[:100])    
    
    print("NearestNeighbors Running time :", time.time() - start)
    
    distances = np.array(list(preds.values()))
    indices = np.array(list(preds.keys()))
    predictions = []

    idx = np.where(distances < threshold)[0]
    ids = indices[idx]
    posting_ids = list(image_name_emb['image_name'].iloc[ids])
    predictions.append(posting_ids)
    
    gc.collect()
    
    
    
    
    return predictions

def get_other_imgembeddings():
    train_embs = np.load('../traindata_embeddings.npy')
    train_path_emb = np.load('../traindata_embeddings_path.npy',allow_pickle=True)
    
    train_path_embeddings = train_path_emb.tolist()
    
    return train_embs, train_path_embeddings

# def get_other_imgembeddings(user_embeddings):
#     train_embs = np.load('traindata_embeddings.npy')
#     print(train_embs.shape)
#     train_path_emb = np.load('traindata_embeddings_path.npy',allow_pickle=True)
    
#     image_embeddings = np.insert(train_embs,0,user_embeddings, axis=0)
#     train_path_embeddings = np.insert(train_path_emb,0,"https://thumbnail.10x10.co.kr/webimage/image/basic600/368/B003684850.jpg", axis=0)
#     train_path_embeddings = train_path_embeddings.tolist()
    
#     return image_embeddings, train_path_embeddings


def run(url):
    user_embeddings = get_image_embeddings(img_url=url)
    
    train_embeddings, train_path_embeddings = get_other_imgembeddings()
    train_path_embeddings = pd.DataFrame(train_path_embeddings, columns=['image_name'])
    
    image_pred = get_image_predictions(user_embeddings, train_embeddings,train_path_embeddings, 0.7)
    image_pred_dict = {url : image_pred[0][:10]} 
    image_pred_dict = json.dumps(image_pred_dict, indent=4, ensure_ascii = False)
    
    print('---'*30)
    print(image_pred_dict)
    return image_pred





        
if __name__ == '__main__':
    pred_group = run("https://thumbnail.10x10.co.kr/webimage/image/basic600/368/B003684850.jpg")