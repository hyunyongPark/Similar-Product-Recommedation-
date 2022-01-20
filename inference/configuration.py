import os 

class Config:
    META_CSV = '/mnt/hdd1/wearly/deep_rec/data/5_features.csv'
    
    DATA_DIR = '/mnt/hdd1/wearly/aim_code/yolov5/data/kfashion/images/all'
    TRAIN_EMB = '/mnt/hdd1/wearly/deep_rec/traindata_embeddings.npy'
    SEED = 123

    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    BATCH_SIZE = 1
    N_FOLDS = 5
    FC_DIM = 512
    
    NUM_WORKERS = 4
    DEVICE = 'cuda:0'
     
    CLASSES = 6000
    SCALE = 30 
    MARGIN = 0.5

    MODEL_NAME = 'tf_efficientnet_b4'
    MODEL_PATH = '../valid_test_tf_efficientnet_b4_30_Weights/tf_efficientnet_b4_29EpochStep_adamw.pt'
    

    

    

    