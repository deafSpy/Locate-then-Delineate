import sys
from loguru import logger
import datetime
import shutil
import glob
import os
import wandb
import torch
import cv2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils.config_based_retrieval import *
from utils.config_handling import *
from utils.callbacks import *
from tools.create_dataset import *
from tools.image_dataclass import *
from tools.image_text_dataclass import *
from engine.engine import *
from matplotlib import pyplot as plt

pl.seed_everything(seed=42)

CONFIG_FOLDER_PATH = "./configs/"

class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = create_model(config)
        self.loss_fn = get_loss_fn(config)
        self.metric_fn = get_metric_fn(config)
        self.config = config

        # basic hyperparameters
        self.num_epochs = config["num_epochs"]
        self.warmup_epochs = config["warmup_epochs"]
        self.weight_decay = config["weight_decay"]
        self.learning_rate = config["learning_rate"]
        self.momentum = config["momentum"]
        self.alpha = config["alpha"]
        self.save_hyperparameters()
    
    def loss(self, preds, y):
        return self.loss_fn(preds, y, self.alpha)

    def metric(self, preds, y):
        return self.metric_fn(preds, y)

    def forward(self, X, text_embed=None, name=None):
        if(text_embed is not None):
            return self.model(X, text_embed)
        else:
            return self.model(X)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    config_path = "configs/qata_inference.yaml"
    # config_path = "configs/mosmed_inference.yaml"
    config = load_config(config_path)
    model_type = ["contextualnet", "unet", "lvit", "mynetwork"]
    
    if config["dataset"] == "qata":
        # models = ["cnet3", "unet3", "lvit_t5", "proposed_bert"]
        models = ["cnet3", "unet3", "lvit_t5", "proposed_t5"]
        models_bert = ["cnet_bert", "unet_bert", "lvit_bert", "proposed_bert"]
    elif config["dataset"] == "ct_lesions":
        models = ["cnet_t5", "unet_t5", "lvit_t5", "proposed_t5"]
        models_bert = ["cnet_t5", "unet_bert", "lvit", "proposed_bert"]
    else:
        print("I messed up")
    
    if config["embedding"] == "bert":
        models = models_bert
        
    _, _, test_dataloader = create_dataset(
            config=config,
            fold=config["fold"],
            img_size=config["img_size"],
            transform=config["transform"],
            num_workers=config["num_workers"],
            batch_size=config["batch_size"],
            dataset_type=config["dataset_type"],
            word_len=config["word_len"],
            # return_test=True
    )
    
    # logging
    logger.info(f"Number of GPUs: {gpus}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"loss: {config['loss']}, metric: {config['metric']}")
    # /ssd_scratch/cvit/shreyu/datasets/qata/output/
    images = []
    masks = []
    frames = []
    names = []
    dices = []

    data = pd.read_csv(os.path.join(config['dataset_path'], config['dataset'], config["text_path"])).to_numpy()
    text = dict(zip(data[:, 1],  data[:, 2]))
    
    for batch in tqdm(test_dataloader):
        X, y, text_embed, name = batch
        print(X.shape, y.shape, text_embed.shape, name[0])
        X = torch.squeeze(X.type(torch.cuda.FloatTensor)).cpu().detach()
        y = torch.squeeze(y.type(torch.cuda.FloatTensor)).cpu().detach()
        print(X.shape, y.shape, text_embed.shape, name[0])
        # text_embed = text_embed.type(torch.cuda.FloatTensor)
        
        masks.append(X)
        frames.append(y)
        names.append(name[0])
        

    for i in range(len(models)):
        model_path = config["dataset"] + "_" + models[i] if i < len(models) else models_bert[i]
        if config["dataset"] == "ct_lesions": 
            model_path = "lesion" + "_" + models[i] if i < len(models) else models_bert[i]
            
                
        
        logger.info(f"Model: {model_path}")

        config["model"] = model_type[i]
        model = LightningModel(config)
        
        checkpoint_path = os.path.join(config['dataset_path'], config["dataset"], "output", model_path)
        logger.info("Getting Results from model")
        testing_model = LightningModel(config).cuda()
        
        state_dict = torch.load(os.path.join(checkpoint_path, "final_model.pth"))
        # print(state_dict)
        testing_model.load_state_dict(state_dict)
        # print(testing_model)
        testing_model.eval()
        
        preds, dice = test2(testing_model, test_dataloader, config, get_metric_fn(config))
        images.append(preds)
        dices.append(dice)
        # print("predsshape", preds.shape)
    
    
    for i in range(len(masks)):
        image_frame = frames[i]
        image_mask = masks[i]
        fig, ax = plt.subplots(1, 6, figsize=(15, 3), dpi=100)
        plt.subplots_adjust(wspace=0.0, top=0.78, bottom=0.18)  
        plt.suptitle(f"Inferences of {names[i]}", fontsize=16)
        
        ax[0].imshow(image_mask, cmap="gray")
        ax[1].imshow(image_frame, cmap="gray")
        ax[2].imshow(images[0][i], cmap="gray")
        ax[3].imshow(images[1][i], cmap="gray")
        ax[4].imshow(images[2][i], cmap="gray")
        ax[5].imshow(images[3][i], cmap="gray")

        vis_type = "X-Ray" if config["dataset"] == "qata" else "CT"
        ax[0].set_title(f"{vis_type} Image")
        ax[1].set_title("Ground Truth Mask")
        ax[2].set_title("ContextualNet")
        ax[3].set_title("UNet")
        ax[4].set_title("LVIT")
        ax[5].set_title("Proposed")

        for a in ax:
            a.axis("off")
            
        for j in range(2,6):
            ax[j].text(0.5, -0.05, f"{dices[j - 2][i]:.3f}", ha='center', va='center', transform=ax[j].transAxes, fontsize=10)


        caption_text = f'$\mathbf{{Text\:Report:}}$ "{text[names[i]]}"'
        plt.figtext(0.5, 0.02, caption_text, ha='center', fontsize=12, wrap=True)

        # Save the figure
        output_path = "./inferences"
        if not config["test"]:
            output_path = os.path.join(config["inference_path"], config["dataset"])
        plt.savefig(os.path.join(output_path, f"inference_{names[i]}"), bbox_inches='tight', pad_inches=0.1)

        # print(images[0][0], np.any(images[0][0]))