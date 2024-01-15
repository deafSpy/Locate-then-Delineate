import torch
import pandas as pd
from utils.schedulers import *
from utils.metrics import *
from loguru import logger
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import torch.nn.functional as F
from skimage import segmentation
import cv2

def configure_optimizers(self):
    if(self.config["optimizer"] == "AdamW"):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    if(self.config["scheduler"] == "LinearWarmupCosineAnnealingLR"):
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.num_epochs)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}

def training_step(self, batch, batch_idx):
    if(len(batch) == 3):
        X, y, _ = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)

        outputs = self.model(X)
    else:
        X, y, text_embed, _ = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        text_embed = text_embed.type(torch.cuda.FloatTensor)

        outputs = self.model(X, text_embed)

    loss = self.loss(outputs, y)

    self.log("train/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    return loss

def validation_step(self, batch, batch_idx):
    if(len(batch) == 3):
        X, y, _ = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)

        outputs = self.model(X)
    else:
        X, y, text_embed, _ = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        text_embed = text_embed.type(torch.cuda.FloatTensor)
        outputs = self.model(X, text_embed)

    val_loss = self.loss(outputs, y)
    outputs = outputs > 0.5
    y = y > 0.5
    val_metric = self.metric(y, outputs)
    self.log("val/metric", val_metric, sync_dist=True, on_step=False, on_epoch=True)
    self.log("val/loss", val_loss, sync_dist=True, on_step=False, on_epoch=True)
    return val_loss

def test(model, test_dataloader, metric_fn, output_dir, save_outputs=True):
    rows = []
    model.cuda()
    model.eval()

    shutil.rmtree(f"{output_dir}/outputs1", ignore_errors=True)
    os.makedirs(f"{output_dir}/outputs1", exist_ok=True)

    shutil.rmtree(f"{output_dir}/outputs2", ignore_errors=True)
    os.makedirs(f"{output_dir}/outputs2", exist_ok=True)

    negatives = []
    small = []
    large = []
    medium = []
    tp, fp, tn, fn = 0, 0, 0, 0

    for batch in tqdm(test_dataloader):
        if(len(batch) == 3):
            X, y, name = batch
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            outputs = model(X)
        else:
            X, y, text_embed, name = batch
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            text_embed = text_embed.type(torch.cuda.FloatTensor)
            outputs = model(X, text_embed)
        
        name = name[0]
        
        outputs = outputs > 0.5
        y = y > 0.5

        test_metric = metric_fn(outputs, y)

        ptx_size = None

        if (y.sum() == 0):
            ptx_size = "negative"
        elif(y.sum() < 375):
            ptx_size = "small"
        elif(y.sum() > 1250):
            ptx_size = "medium"
        else:
            ptx_size = "large"

        row = {
            "name": name,
            "ptx_size": ptx_size,
            "Adjusted rand idx": calc_AdjustedRandIndex(y, outputs),
            "Dice": dice_metric(y, outputs),
            "Confusion Matrix":calc_ConfusionMatrix(y, outputs),
            "Accuracy": calc_Accuracy_Sets(y, outputs),
            "Hausdorff dist.": calc_AverageHausdorffDistance(y, outputs),
            "AUC": calc_AUC_trapezoid(y, outputs),
            "Sensitivity": calc_Sensitivity_Sets(y, outputs),
            "Precision": calc_Precision_Sets(y, outputs),
            "Specificity": calc_Specificity_Sets(y, outputs),
        }
        rows.append(row)
        
        #print(outputs.shape)
        preds = outputs.squeeze().cpu().detach()
        preds = preds.numpy()
        
        preds = np.sum(preds, 0)/ preds.shape[0]
        
        preds = preds + 0
        preds *= 255

        '''
        preds2 = outputs2.squeeze().cpu().detach()
        preds2 = preds2.numpy()
        
        preds2 = np.sum(preds2, 0)/ preds2.shape[0]
        
        preds2 = preds2 + 0
        preds2 *= 255

        preds1 = cv2.resize(preds1, (256, 256))
        preds2 = cv2.resize(preds2, (256, 256))
        '''
        preds = np.array(preds, dtype='uint8')
        
        if(y.sum() == 0 and preds.sum() == 0):
            tn = tn + 1
        elif(y.sum() == 0 and preds.sum() > 0):
            fp = fp + 1
        elif(y.sum() > 0 and preds.sum() == 0):
            fn = fn + 1
        else:
            tp = tp + 1
        
        if(save_outputs):
            cv2.imwrite(f"{output_dir}/outputs/{name}.png", preds)
            #cv2.imwrite(f"{output_dir}/outputs2/{name}.png", preds2)
        
        # insights
        if(y.sum() == 0):
            negatives.append(test_metric.item())
        elif(y.sum() < 375):
            small.append(test_metric.item())
        elif(y.sum() < 1250):
            medium.append(test_metric.item())
        else:
            large.append(test_metric.item())

    average_negative_dice = np.mean(negatives)
    average_small_dice = np.mean(small)
    average_medium_dice = np.mean(medium)
    average_large_dice = np.mean(large)

    average_positive_dice = (np.sum(small)+np.sum(medium)+np.sum(large)) / (len(small)+len(medium)+len(large))
    average_med_large_dice = (np.sum(medium)+np.sum(large)) / (len(medium)+len(large))
    overall_dice = (np.sum(negatives)+np.sum(small)+np.sum(medium)+np.sum(large)) / (len(small)+len(medium)+len(large)+len(negatives))

    classification_accuracy = (tp + tn)/(tp + fp + tn + fn)

    insights= f"""Thresholds for sub categories: (375, 1250)
    Average Dice: {overall_dice},
    Average Positives Dice: {average_positive_dice},
    Average Negatives Dice: {average_negative_dice},
    Average Small Pneumothorax Dice: {average_small_dice},
    Average Medium Pneumothorax Dice: {average_medium_dice},
    Average Large Pneumothorax Dice: {average_large_dice},
    Average Dice for Medium and Large Pneumothorax: {average_med_large_dice},
    [TP, FP, TN, FN] : [{tp}, {fp}, {tn}, {fn}],
    Classication Accuracy : {classification_accuracy},
    Number of images (s, m, l, n): [{len(small)}, {len(medium)}, {len(large)}, {len(negatives)}],
    """
    
    with open(f'{output_dir}/insights.txt', 'w') as f:
        f.write(insights)
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/eval_scores.csv") 
