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
from pathlib import Path
from transformers import AutoTokenizer
from bert_embedding import BertEmbedding
import pandas as pd
from os.path import dirname

import gc
gc.set_threshold(0)

class ImageTextDataClass(data.Dataset):
    def __init__(self, config, files, max_len=150, mode="val", img_size=256, transform=False):
        super(ImageTextDataClass, self).__init__()
        self.img_files = files
        print(len(self.img_files))
        self.is_transform = transform
        self.img_size = img_size
        self.transforms = image_transforms(self.img_size)
        self.mode = mode
        self.resize = transforms.Resize((img_size, img_size))
        self.max_len = max_len
        # self.tokenizer = AutoTokenizer.from_pretrained(config["t5_path"])
        
        data = pd.read_csv(os.path.join(config['dataset_path'], config['dataset'], config["text_path"])).to_numpy()
        self.df = data
        self.text = dict(zip(data[:, 1],  data[:, 2]))
        # print(data[:, 1])
        # print("THIS IS A TEXT", len(self.text), self.text['sub-S09934_ses-E17062_run-1_bp-chest_vp-ap_dx.png'])
        self.bert_embedding = BertEmbedding(max_seq_length=10)

        self.config = config
        # if('stage1' in config):
        #     self.df = pd.read_csv(os.path.join('qinfo', config['stage1']+'.csv'))

    def __getitem__(self, index):
        
        # print("GETITEM")
        img = cv.imread(self.img_files[index])
        # img = np.uint16(img_raw)
        # img = normalise(img_raw)
        mask = cv.cvtColor(cv.imread(self.img_files[index].replace("frames", "masks")), cv.COLOR_BGR2GRAY)
        mask = np.uint8(mask)
        
        # augmentation
        if self.is_transform and self.mode == "train":
            # img = Image.fromarray(img).convert("RGB")
            img = np.array(img)
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        image = Image.fromarray(img).convert("RGB")
        image = self.resize(image)
        image = np.array(image) / 255
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(np.array(image[0, :, :]),0)
        
        segmentation_mask = Image.fromarray(mask)
        segmentation_mask = self.resize(segmentation_mask)
        mask = np.array(np.expand_dims(np.array(segmentation_mask), 0) / 255, dtype='uint8')

        # text
        # print(os.path.basename(self.img_files[index]))
        text = self.text[os.path.basename(self.img_files[index])]

        # inputs = self.tokenizer.encode_plus(
        #     text,
        #     None,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     padding= 'max_length',
        #     truncation='longest_first',
        #     return_token_type_ids=True
        # )
        # ids = inputs['input_ids']
        # att_mask = inputs['attention_mask']
        # token_type_ids = inputs["token_type_ids"]

        text = text.split("\n")
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 10:
            text = text[:10, :]
        else:
            tmp = np.zeros((10, 768))
            if (text.shape[0] > 0):
                tmp[:text.shape[0], :] = text
            text = tmp

        # if(self.config["model"] == 'lvit' or self.config["embedding"] == "bert"):
        # if (self.config["embedding"] == "bert"):
        # # if(self.config["embedding"] == "bert" and self.config["model"] != "contextualnet"):
        #     print("Using inbuilt bert")
        #     return image, mask, text, os.path.basename(self.img_files[index])
        # print("Using precomputed bert")
        #loading precomputed text embeddings to save training time
        
        # t5_embeddings = torch.load('/ssd_scratch/cvit/shreyu/datasets/ptx-textseg-dataset/candid_ptx_dataset/encoded_embeddings/'+os.path.basename('0.1.18.307998.23.3.8.6.07708604033.4278452263112.0.pt'))
        # t5_embeddings = torch.load('/ssd_scratch/cvit/shreyu/datasets/ptx-textseg-dataset/candid_ptx_dataset/encoded_embeddings/'+os.path.basename('9.9.94.760843.29.4.4.3.35136456719.8326270478513.9.pt'))
        # print(os.path.basename(self.img_files[index])+'.pt')
        embeddings = torch.load(os.path.join(f'/ssd_scratch/cvit/shreyu/datasets/{self.config["dataset"]}/{self.config["embedding"]}_encoded_embeddings/', os.path.basename(self.img_files[index]).replace("png", "pt")))
        
        # if self.config["model"] == "contextualnet":
        #     embeddings = torch.load(os.path.join(f'/ssd_scratch/cvit/shreyu/datasets/{self.config["dataset"]}/t5_encoded_embeddings/', os.path.basename(self.img_files[index]).replace("png", "pt")))
            
        print("emb", embeddings.shape)
        # exit(0)
        # t5_embeddings = None
        # if(self.config["model"] == "mynetwork"):
        #     row = self.df.loc[self.df['name'] == os.path.basename(self.img_files[index])]
        #     if(self.config['quad_num'] == 4):
        #         quad_info = [row['q1'], row['q2'],row['q3'],row['q4']]
        #         quad_info = [0,0,0,0]
        #     elif(self.config['quad_num'] == 6):
        #         quad_info = [row['h1'],row['h2'],row['h3'],row['h4'],row['h5'],row['h6']]

        #     return image, mask, np.array(quad_info), os.path.basename(self.img_files[index])

        # return image, mask, text_token, os.path.basename(self.img_files[index])
        return image, mask, embeddings, os.path.basename(self.img_files[index])
        
    def __len__(self):
        # print("GOTLENGTH")
        return len(self.img_files)
