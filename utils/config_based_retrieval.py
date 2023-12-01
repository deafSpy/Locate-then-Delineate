from benchmarks.unet import UNET
from benchmarks.contextual_net import CONTEXTUALNET
from utils.losses import *
from utils.metrics import *
from benchmarks.LViT import *
from benchmarks.cpam import *
from model.network import *
import torchvision

def create_model(config):
    if(config["model"] == "unet"):
        return UNET(config)
    elif(config["model"] == "contextualnet"):
        return CONTEXTUALNET(config)
    elif(config["model"] == "lvit"):
        return LViT(config)
    elif(config["model"] == "cpam"):
        return UnetCPAM(config)
    elif(config["model"] == "mynetwork"):
        return MyNetwork(config)
    
def get_loss_fn(config):
    if(config["loss"] == "dice_loss"):
        return dice_loss
    elif(config["loss"] == "bce_dice"):
        return bce_dice
        
def get_metric_fn(config):
    if(config["metric"] == "dice_metric"):
        return dice_metric
    if(config["metric"] == "classification_accuracy"):
        return classification_accuracy
