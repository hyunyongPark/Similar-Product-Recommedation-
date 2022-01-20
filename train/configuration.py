import os 

class Config:
    
    DATA_DIR = '/mnt/hdd1/wearly/aim_code/yolov5/data/kfashion/images/all'
    TRAIN_CSV = '/mnt/hdd1/wearly/deep_rec/data/train_csv/150limits.csv'
#     TRAIN_CSV = '/mnt/hdd1/wearly/deep_rec/data/train_csv/50limits_csv'
    SEED = 123
    SAVE_NAME = 'valid_test'

    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    EPOCHS = 30 #15  # Try 15 epochs
    BATCH_SIZE = 64
    N_FOLDS = 5
    
    NUM_WORKERS = 4
    DEVICE = 'cuda:0'
    
    TYP = 'train'
    
    #HEIGHT=512 #for augmentation
    #WIDTH=512
    
    CLASSES = 6000#1359
    SCALE = 30 
    MARGIN = 0.5

    MODEL_NAME =  'tf_efficientnet_b4'
    FC_DIM = 512
    
    #LR
    LR_START = 2e-5
    
    weight_decay = 0.0
    optimizer_name = 'adam'
    