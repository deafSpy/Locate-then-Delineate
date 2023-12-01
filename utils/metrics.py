import torch
from sklearn.metrics import *
import numpy as np
import scipy.ndimage as ndimage
from torchmetrics.classification import Dice, BinaryAccuracy, MulticlassAccuracy

def classification_accuracy(target, pred):
    accuracy = BinaryAccuracy().to('cuda')
    x = accuracy(pred[:,0], target[:,0])
    return x

def dice_metric(target, pred, smooth_nr=1e-5, smooth_dr=1e-5):
    dice = Dice(zero_division = 1, ignore_index = 0).to('cuda')
    return dice(pred, target)

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/accuracy.py
def calc_AdjustedRandIndex(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    gt = np.equal(truth, c).flatten()
    pd = np.equal(pred, c).flatten()
    # Compute ARI via scikit-learn
    ari = adjusted_rand_score(gt, pd)
    # Return ARI score
    return np.float64(ari)

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/accuracy.py
def calc_Accuracy_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Calculate Accuracy
    acc = (np.logical_and(pd, gt).sum() + \
           np.logical_and(not_pd, not_gt).sum()) / gt.size
    # Return computed Accuracy
    return acc

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/auc.py
def calc_AUC_trapezoid(truth, pred, c=1, **kwargs):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Compute AUC
    if (fp+tn) != 0 : x = fp/(fp+tn)
    else : x = 0.0
    if (fn+tp) != 0 : y = fn/(fn+tp)
    else : y = 0.0
    auc = 1 - (1/2)*(x + y)
    # Return AUC
    return auc

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/confusion_matrix.py
def calc_ConfusionMatrix(truth, pred, c=1, dtype=np.int64, **kwargs):
    # Obtain predicted and actual condition
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    # Compute Confusion Matrix
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    # Convert to desired numpy type to avoid overflow
    tp = tp.astype(dtype)
    tn = tn.astype(dtype)
    fp = fp.astype(dtype)
    fn = fn.astype(dtype)
    # Return Confusion Matrix
    return tp, tn, fp, fn

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/hausdorff.py
def border_map(binary_img):
    """
    Creates the border for a 3D or 2D image
    """
    ndims = binary_img.ndim
    binary_map = np.asarray(binary_img, dtype=np.uint8)

    if ndims == 2:
        left = ndimage.shift(binary_map, [-1, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, -1], order=0)
        cumulative = left + right + superior + inferior
        ndir = 4
    elif ndims == 3:
        left = ndimage.shift(binary_map, [-1, 0, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0, 0], order=0)
        anterior = ndimage.shift(binary_map, [0, 1, 0], order=0)
        posterior = ndimage.shift(binary_map, [0, -1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, 0, -1], order=0)
        cumulative = left + right + anterior + posterior + superior + inferior
        ndir = 6
    else:
        raise RuntimeError(f'Image must be of 2 or 3 dimensions, got {ndims}')

    border = ((cumulative < ndir) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    border_ref = border_map(ref)
    border_seg = border_map(seg)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg

def calc_AverageHausdorffDistance(truth, pred, c=1, **kwargs):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    # Obtain sets with associated class
    ref = np.equal(truth, c)
    seg = np.equal(pred, c)
    ref = np.squeeze(ref, axis=0)
    seg = np.squeeze(seg, axis=0)
    # Compute AHD
    ref_border_dist, seg_border_dist = border_distance(ref, seg)
    hausdorff_distance = np.max([np.max(ref_border_dist),
                                 np.max(seg_border_dist)])
    # Return AHD
    return hausdorff_distance

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/precision.py
def calc_Precision_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    # Calculate precision
    if pd.sum() != 0 : prec = np.logical_and(pd, gt).sum() / pd.sum()
    else : prec = 0.0
    # Return precision
    return prec

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/sensitivity.py
def calc_Sensitivity_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    gt = np.equal(truth, c)
    pd = np.equal(pred, c)
    # Calculate sensitivity
    if gt.sum() != 0 : sens = np.logical_and(pd, gt).sum() / gt.sum()
    else : sens = 0.0
    # Return sensitivity
    return sens

#https://github.com/frankkramer-lab/miseval/blob/master/miseval/specificity.py
def calc_Specificity_Sets(truth, pred, c=1, **kwargs):
    # Obtain sets with associated class
    truth = truth.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    not_gt = np.logical_not(np.equal(truth, c))
    not_pd = np.logical_not(np.equal(pred, c))
    # Calculate specificity
    if (not_gt).sum() != 0:
        spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
    else : spec = 0.0
    # Return specificity
    return spec
