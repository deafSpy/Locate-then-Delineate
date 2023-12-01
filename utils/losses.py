from utils.metrics import *
import torch.nn.functional as F
from utils.generalized_dice_loss import GeneralizedDiceLoss
import torch.nn.functional as F

gdl = GeneralizedDiceLoss(include_background=False, sigmoid=True)

def bce_dice(pred, target, alpha):
    loss =  alpha*F.binary_cross_entropy_with_logits(pred, target) + (1-alpha)*gdl(pred, target)
    return loss

def bce(pred, target, alpha):
    loss =  F.binary_cross_entropy_with_logits(pred, target)
    return loss