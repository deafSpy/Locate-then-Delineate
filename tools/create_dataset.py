import numpy as np
import pandas as pd
from tools.image_dataclass import *
from tools.image_text_dataclass import *
from torch.utils.data import DataLoader
import os
from loguru import logger
from sklearn.model_selection import train_test_split
import cv2 as cv

import gc
gc.set_threshold(0)

def create_dataset(config, fold, img_size, transform, num_workers, batch_size, dataset_type="image", word_len=150):
    df_folds = pd.read_csv(os.path.join(os.getcwd(), "tools", config["folds_file"]))
    train_files = df_folds[f"fold_{fold}_train"].tolist()
    test_files = df_folds[f"fold_{fold}_test"].tolist()
    val_files = df_folds[f"fold_{fold}_val"].tolist()
    test_files = np.array(test_files)
    train_files = np.array(train_files)
    val_files = np.array(val_files)
    
    # print(os.path.join(os.getcwd(), "tools", "folds.csv"))
    # exit(1)
    # print("cd debug", df_folds, train_files, val_files)

    test_files = test_files[np.where(test_files != '-1')]
    test_files = [os.path.join(config['dataset_path'], config['dataset'], "dicom_files", test_file) for test_file in test_files]
    train_files = train_files[np.where(train_files != '-1')]
    train_files = [os.path.join(config['dataset_path'], config['dataset'], "dicom_files", train_file) for train_file in train_files]
    val_files = val_files[np.where(val_files != '-1')]
    val_files = [os.path.join(config['dataset_path'], config['dataset'], "dicom_files", val_file) for val_file in val_files]

    
    if(dataset_type == "image"):
        train_data = ImageDataClass(config, train_files, mode="train", img_size=img_size, transform=transform)
        val_data = ImageDataClass(config, val_files, img_size=img_size, transform=transform)
        test_data = ImageDataClass(config, test_files, img_size=img_size, transform=transform)
        logger.info("Datasets prepared")
    elif(dataset_type == "image-text"):
        # print("train")
        train_data = ImageTextDataClass(config, train_files, mode="train", img_size=img_size, transform=transform, max_len=word_len)
        # print("val")
        val_data = ImageTextDataClass(config, val_files, mode="val", img_size=img_size, transform=transform, max_len=word_len)
        # print("ttest")
        test_data = ImageTextDataClass(config, test_files, mode="val", img_size=img_size, transform=transform, max_len=word_len)
        logger.info("Datasets prepared")

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Dataloaders created")

    return train_dataloader, val_dataloader, test_dataloader

