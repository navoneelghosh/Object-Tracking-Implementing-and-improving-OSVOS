from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.

Modified by: Navoneel Ghosh
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)

# User defined parameters
# seq_name = "car-shadow"
# gpu_id = 0
# train_model = False
# result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
# kalman_result_path = os.path.join('DAVIS', 'KalmanResults', 'Segmentations', '480p', 'OSVOS', seq_name)

def train_and_test_osvos(seq_name, gpu_id, result_path, train_model):
    # Train parameters
    parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
    logs_path = os.path.join('models', seq_name)
    max_training_iters = 2000

    # Define Dataset
    test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
    test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
    if train_model:
        train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                    os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
        dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
    else:
        dataset = Dataset(None, test_imgs, './')

    # Train the network
    if train_model:
        # More training parameters
        learning_rate = 1e-8
        save_step = max_training_iters
        side_supervision = 3
        display_step = 10
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                    save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

    # Test the network
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
            osvos.test(dataset, checkpoint_path, result_path)

