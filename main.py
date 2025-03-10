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

import gc
gc.set_threshold(0)

np.random.seed(seed=42) 
torch.manual_seed(seed=42) 
torch.cuda.manual_seed(seed=42) 
torch.cuda.manual_seed_all(seed=42) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 
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
    config_path = sys.argv[1]
    config = load_config(config_path)
    print(config)
    if(config["debug"]):
        config["wandb_run_name"] = "**DEBUG** " + config["wandb_run_name"]

    # logging
    logger.info(f"Number of GPUs: {gpus}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Model: {config['model']}, loss: {config['loss']}, metric: {config['metric']}")
    logger.info(f"DEBUG: {config['debug']}")

    wandb_logger = WandbLogger(name=config["wandb_run_name"] + "-" + str(datetime.datetime.now()), project=config["wandb_project"], log_model="false")

    train_dataloader, val_dataloader, test_dataloader = create_dataset(
                                    config=config,
                                    fold=config["fold"], # unused
                                    img_size=config["img_size"],
                                    transform=config["transform"],
                                    num_workers=config["num_workers"],
                                    batch_size=config["batch_size"],
                                    dataset_type=config["dataset_type"],
                                    word_len=config["word_len"]
    )

    
    LightningModel.training_step = training_step
    LightningModel.validation_step = validation_step
    LightningModel.configure_optimizers = configure_optimizers
    
    model = LightningModel(config)
    
    shutil.rmtree(os.path.join(config['dataset_path'], config["dataset"], "output", config['final_dir_name']), ignore_errors=True)
    os.makedirs(os.path.join(config['dataset_path'], config["dataset"], "output", config['final_dir_name']), exist_ok=True)
    
    checkpoint_path = os.path.join(config['dataset_path'], config["dataset"], "output", config['final_dir_name'], "checkpoints")
    print("checkpoint_path", checkpoint_path)
    
    shutil.copy(os.path.join(CONFIG_FOLDER_PATH, config['config_name']), os.path.join(config['dataset_path'], config["dataset"], "output", config['final_dir_name'], "config.yaml"))
    
    overall_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        monitor="train/loss",
        save_top_k=config["save_top_k"],
        mode="min")

    val_loss_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best_val_loss",
        monitor="val/loss",
        save_top_k=1,
        mode="min")

    val_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="best_val_acc",
        monitor="val/metric",
        save_top_k=1,
        mode="max")

    max_epochs = 101

    callbacks = [overall_checkpoint_callback, val_loss_checkpoint_callback, val_acc_checkpoint_callback]
    if(config["debug"] == False):
        max_epochs = config["num_epochs"]

    logger.info(f'gpus: {gpus}')
    # print("HIHIHIHI", any(p.requires_grad for p in model.parameters()))
    # exit(0)

    trainer = pl.Trainer(
        devices=gpus, 
        accelerator="gpu", 
        strategy = DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger, 
        callbacks=callbacks, 
        max_epochs=max_epochs,
        deterministic=True
    )
    
    logger.info("Beginning to train")
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
                )
    
    torch.save(model.state_dict(), os.path.join(config['dataset_path'],  config["dataset"], "output", config['final_dir_name'], "final_model.pth"))

    logger.info("Testing model")
    testing_model = LightningModel(config).load_from_checkpoint(f"{checkpoint_path}/best_val_loss.ckpt")
    test(testing_model, test_dataloader, config, get_metric_fn(config), os.path.join(config['dataset_path'], config["dataset"], "output", config['final_dir_name']), save_outputs=True)
    wandb.finish()
