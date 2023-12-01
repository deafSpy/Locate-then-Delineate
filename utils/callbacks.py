import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.with_text = True
        if(len(val_samples) == 3):
            self.val_imgs, self.val_masks, self.names = val_samples
            self.with_text = False
        else:
            self.val_imgs, self.val_masks, self.val_text_embeds, self.names = val_samples
            self.val_text_embeds = self.val_text_embeds[:num_samples]

        self.val_imgs = self.val_imgs[:num_samples]
        self.val_masks = self.val_masks[:num_samples]
        self.names = self.names[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.type(torch.cuda.FloatTensor)
        val_masks = self.val_masks.type(torch.cuda.FloatTensor)
        if(self.with_text):
            val_text_embeds = self.val_text_embeds.type(torch.cuda.FloatTensor)
            
            ##LViT
            #val_ids = self.val_ids.type(torch.cuda.FloatTensor)
            #val_att_masks = self.val_att_masks.type(torch.cuda.FloatTensor)
            outputs = pl_module(val_imgs, val_text_embeds)
        else:
            outputs = pl_module(val_imgs)

        outputs = outputs.squeeze()
        final_pred = (outputs > 0.5) + 0
        outputs = outputs - torch.min(outputs)
        outputs = outputs/torch.max(outputs)
        val_imgs = val_imgs[:, 0, :, :]
        val_masks = val_masks.squeeze()
        concatenated_imgs = torch.cat((val_imgs, val_masks, outputs, final_pred), dim=1)
        trainer.logger.experiment.log({
           "examples": [wandb.Image(concatenated_img, caption=f"File: {name}") 
                           for concatenated_img, name in zip(concatenated_imgs, self.names)],
           "global_step": trainer.global_step
           })
