import sys
from loguru import logger
import datetime
import shutil
import glob
import os
import wandb
import torch
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

pl.seed_everything(seed=42)

SCRATCH_FOLDER_PATH = "/scratch/loki"
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
    config_path = sys.argv[1]
    config = load_config(config_path)

    # logging
    logger.info(f"Number of GPUs: {gpus}")
    logger.info(f"Dataset: {config['dataset']}, fold: {config['fold']}")
    logger.info(f"Model: {config['model']}, loss: {config['loss']}, metric: {config['metric']}")
    logger.info(f"DEBUG: {config['debug']}")

    train_dataloader, val_dataloader, test_dataloader = create_dataset(
            config,
            fold=config["fold"],
            img_size=config["img_size"],
            transform=config["transform"],
            num_workers=config["num_workers"],
            batch_size=config["batch_size"],
            dataset_type=config["dataset_type"],
            word_len=config["word_len"]
    )
    
    model = LightningModel(config)

    os.makedirs(os.path.join(SCRATCH_FOLDER_PATH, config['final_dir_name']), exist_ok=True)
    
    checkpoint_path = os.path.join(SCRATCH_FOLDER_PATH, config['final_dir_name'], "checkpoints")
    shutil.copy(os.path.join(CONFIG_FOLDER_PATH, config['config_name']), os.path.join(SCRATCH_FOLDER_PATH, config['final_dir_name'], "config.yaml"))

    logger.info("Testing model")
    testing_model = LightningModel(config).load_from_checkpoint(f"{checkpoint_path}/best_val_loss.ckpt")
    test(testing_model, test_dataloader, get_metric_fn(config), os.path.join(SCRATCH_FOLDER_PATH, config['final_dir_name']), save_outputs=True)
