from re import I
import cv2 as cv
import numpy as np
from torch.utils import data
import torch
from torchvision import transforms
import pydicom
from PIL import Image
import os
from tools.image_preprocessing import *
from loguru import logger

class ImageDataClass(data.Dataset):
    def __init__(self, config, files, mode="val", img_size=256, transform=False):
        super(ImageDataClass, self).__init__()
        self.img_files = files
        self.is_transform = transform
        self.img_size = img_size
        self.transforms = image_transforms(self.img_size)
        self.mode = mode
        self.resize = transforms.Resize((img_size, img_size))
        self.config = config
        
    def __getitem__(self, index):
        img_raw = pydicom.dcmread(self.img_files[index]).pixel_array
        img = normalise(img_raw)
        mask = cv.cvtColor(cv.imread(self.img_files[index].replace("dicom_files", "masks")+".jpg"), cv.COLOR_BGR2GRAY)
        mask = np.uint8(mask)
        
        # augmentation
        if self.is_transform and self.mode == "train":
            img = Image.fromarray(img).convert("RGB")
            img = np.array(img)
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        image = Image.fromarray(img).convert("RGB")
        image = self.resize(image)
        image = np.array(image) / 255
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(np.array(image[0,:,:]),0)
        segmentation_mask = Image.fromarray(mask)
        segmentation_mask = self.resize(segmentation_mask)
        mask = np.array(np.expand_dims(np.array(segmentation_mask), 0) / 255, dtype='uint8')

        return image, mask, [os.path.basename(self.img_files[index]), np.sum(mask)]
    
    def __len__(self):
        return len(self.img_files)
