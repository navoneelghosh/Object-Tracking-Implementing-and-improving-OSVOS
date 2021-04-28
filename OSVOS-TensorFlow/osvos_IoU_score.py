"""
Navoneel Ghosh

This is an IoU Score implementation between the segmentation result from the OSVOS and the annotations,
performed on data from DAVIS 2016 dataset.

OSVOS is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017 
Please consider citing the original paper if you use this code.

"""

import os
import sys
import numpy as np


def mean_iou_score(annotation_imgs, result_imgs,show_per_frame_iou):
    i = 1
    iou_score_sum = 0
    for a_img, r_img in zip(annotation_imgs, result_imgs):
        intersection = np.logical_and(a_img, r_img)
        union = np.logical_or(a_img, r_img)
        iou_score = np.sum(intersection) / np.sum(union)
        if show_per_frame_iou:
            print("Frame : "+str(i-1)+" score : "+str(iou_score))
            
        iou_score_sum += iou_score
        i += 1
    
    mean_iou_score = iou_score_sum/i
    print("Mean IOU Score : "+str(mean_iou_score))
