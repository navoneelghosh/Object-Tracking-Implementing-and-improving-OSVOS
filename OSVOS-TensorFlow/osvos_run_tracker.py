"""
Navoneel Ghosh

Runnable tracker file that implements OSVOS tracker, Kalman filtering and IoU scores.

OSVOS is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017 
Please consider citing the original paper if you use this code.

"""

import os
import sys
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)
import random
import uuid
import numpy as np
import cv2
import glob
from osvos_train_test import train_and_test_osvos
from osvos_IoU_score import mean_iou_score
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from osvos_kalman_tracker import item

if __name__ == '__main__':
    # User defined parameters
    seq_name = "car-shadow"
    gpu_id = 0
    train_model = False
    result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
    kalman_result_path = os.path.join('DAVIS', 'KalmanResults', 'Segmentations', '480p', 'OSVOS', seq_name)
    og_img_path = os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)

    annotation_filenames = glob.glob(os.path.join('DAVIS', 'Annotations', '480p', seq_name, '*.png'))
    annotation_filenames.sort()
    annotation_imgs = [cv2.imread(img,0) for img in annotation_filenames]

    og_img_filenames = glob.glob(os.path.join(og_img_path, '*.jpg'))
    og_img_filenames.sort()
    og_img = [cv2.imread(img) for img in og_img_filenames]
    
    result_filenames = glob.glob(os.path.join(result_path, '*.png'))
    # print(len(annotation_filenames))
    # print(len(result_filenames))
    # Train and test or just test, depending on the value of "train_model"
    if len(result_filenames)==0:
        train_and_test_osvos(seq_name, gpu_id, result_path, train_model)
        result_filenames.sort()
        result_imgs = [cv2.imread(img,0) for img in result_filenames]
    else:
        result_filenames.sort()
        result_imgs = [cv2.imread(img,0) for img in result_filenames]
    
    # Find contours and reduce noise
    tracker=None
    overlay_color = [255, 0, 0]
    transparency = 0.6
    for img_p,frame in zip(result_imgs,og_img):
        segmentationMask=img_p
        contours, hierarchy =cv2.findContours(segmentationMask,1, 2)
        maxArea=None
        index=-1
        for i,c in enumerate(contours):
            if maxArea is None or cv2.contourArea(c)>cv2.contourArea(maxArea):
                maxArea=c
        newSegmentationMask=np.zeros_like(segmentationMask,dtype=np.uint8)
        cv2.drawContours(newSegmentationMask,[maxArea],-1,(255,255,255),thickness=cv2.FILLED)

        # Write new segmentation mask to file
        # TODO

        # Show result after adding tracking ID and Kalman tracker
        newSegmentationMask = newSegmentationMask//np.max(newSegmentationMask)
        frame[:, :, 0] = (1 - newSegmentationMask) * frame[:, :, 0] + newSegmentationMask * (overlay_color[0]*transparency + (1-transparency)*frame[:, :, 0])
        frame[:, :, 1] = (1 - newSegmentationMask) * frame[:, :, 1] + newSegmentationMask * (overlay_color[1]*transparency + (1-transparency)*frame[:, :, 1])
        frame[:, :, 2] = (1 - newSegmentationMask) * frame[:, :, 2] + newSegmentationMask * (overlay_color[2]*transparency + (1-transparency)*frame[:, :, 2])
        M=cv2.moments(maxArea)
        x,y,w,h = cv2.boundingRect(maxArea)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        loc=np.array([cX,cY])
        if tracker is None:
            tracker=item(loc)
        else:
            path=tracker.update(loc)
            for i in range(len(path) - 1):
                cv2.circle(frame, tuple(path[i].astype(np.int32)), 2, tracker.color)
                cv2.circle(frame, tuple(path[i + 1].astype(np.int32)), 2, tracker.color)
                cv2.line(frame, tuple(path[i].astype(np.int32)), tuple(path[i + 1].astype(np.int32)), tracker.color,
                         thickness=2)
        cv2.imshow('frame',frame)
        cv2.waitKey(300)


    if len(result_filenames)==0:
        print("Results not found!")
    else:
        mean_iou_score(annotation_imgs,result_imgs)
        

    