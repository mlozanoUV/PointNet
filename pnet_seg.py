"""
Title: Point cloud segmentation with PointNet
Author: [Soumik Rakshit](https://github.com/soumik12345), [Sayak Paul](https://github.com/sayakpaul)
Date created: 2020/10/23
Description: Implementation of a PointNet-based model for segmenting point clouds.

Last modified: 2021/11/26
Author: Miguel Lozano (commlab.uv.es)
"""

import os, sys
import json
import random
import numpy as np
import pandas as pd
#from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5, load_weights_from_hdf5_group
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

        
from pnet_vis import visualize_data, visualize_single_point_cloud
from pnet_aux import convert, generate_dataset, get_shape_segmentation_model, run_experiment
import argparse


VAL_SPLIT   = 0.2
BATCH_SIZE  = 1
EPOCHS      = 10
INITIAL_LR  = 1e-03
LABELS      = [0, 1]
NCLASSES    = len(LABELS)
nfiles      = 0           # global, para los nombres de ficheros

if __name__ == '__main__': 

  parser = argparse.ArgumentParser(description='Point Net for Atria Segmentation')
  parser.add_argument('-m','--mode', help='train/test/convert/show (modes)', required=True)
  parser.add_argument('-p','--points', help='Points Cloud File (.npy)', required=False)
  parser.add_argument('-w','--weigths', help='Model Weigths', required=False)
  parser.add_argument('-f','--nfiles', help='Num files to convert', required=False)
  parser.add_argument('-n','--npoints', help='N points / model', required=False)
  parser.add_argument('-v','--val', help='Validation cloud points', required=False)
  
  args = vars(parser.parse_args())
  mode = args['mode']
  
  if args['nfiles'] != None:
    nfiles = int(args['nfiles'])
  if args['npoints'] != None:
    npoints = int(args['npoints'])

  if args['points'] != None:
    pc_file = args['points']
    
  if args['val'] != None:
    val_file = args['val']
  
  if args['weigths'] != None:
    weigths_file = args['weigths']
 
  if mode == 'convert':
    convert(nfiles, npoints)
    exit()
  else:  
    Lp = np.load(pc_file)
    print(pc_file, Lp.shape)
    mask_pc_file = str(np.copy(pc_file)).replace("atria_", "atria_mask_")
    Lm = np.load(mask_pc_file)
    npoints = len(Lp)

    point_clouds = np.copy(Lp)
    point_cloud_labels, all_labels = np.copy(Lm), np.copy(Lm)
    print(Lp.shape, Lm.shape)
    print("Fin load ....... loaded ")
    #visualize_data(point_clouds[0], all_labels[0], LABELS)
        
    if mode == 'show':
      for i in range(len(point_clouds)):
        visualize_data(point_clouds[i], all_labels[i], LABELS)
        
      exit()
 
    """
    ### Preprocessing

    Note that all the point clouds that we have loaded consist of a variable number of points,
    which makes it difficult for us to batch them together. In order to overcome this problem, we
    randomly sample a fixed number of points from each point cloud. We also normalize the
    point clouds in order to make the data scale-invariant.
    """
    Lpc, Llabs = [],[]
    for index in tqdm(range(len(point_clouds))):
        current_point_cloud = point_clouds[index]
        current_label_cloud = point_cloud_labels[index]
        current_labels = all_labels[index]
        
        # Normalizing sampled point cloud.
        center_point_cloud  = current_point_cloud  - np.mean(current_point_cloud, axis=0)
        norm_point_cloud    = center_point_cloud / np.max(np.linalg.norm(center_point_cloud, axis=1))
        
        Lpc.append(norm_point_cloud)
        #point_clouds[index] = norm_point_cloud
        
        #point_cloud_labels[index] = current_label_cloud
        Llabs.append(current_label_cloud)
        all_labels[index] = current_labels
        #visualize_points(Lpc[index])

    NUM_SAMPLE_POINTS = Lpc[0].shape[0]

    point_clouds = Lpc
    point_cloud_labels = Llabs
    #visualize_data(point_clouds[0], all_labels[0])

    for cl in point_cloud_labels:  
        print("Point-Class distr:", (cl ==0).sum()/len(cl), ((cl == 1).sum())/len(cl), (cl == 2).sum()/len(cl))
    
    point_cloud_labels = []
    # Apply one-hot encoding to the dense label representation.
    for case in Llabs:
      label_data = keras.utils.to_categorical(case, num_classes= 3)
      point_cloud_labels.append(label_data)
    
    if mode == 'train':

        split_index             = int(len(point_clouds) * (1 - VAL_SPLIT))
        train_point_clouds      = point_clouds[:split_index]
        train_label_cloud       = point_cloud_labels[:split_index]
        total_training_examples = len(train_point_clouds)

        val_point_clouds        = point_clouds[split_index:]
        val_label_cloud         = point_cloud_labels[split_index:]

        print("Num train point clouds:", len(train_point_clouds))
        print("Num train point cloud labels:", len(train_label_cloud))
        print("Num val point clouds:", len(val_point_clouds))
        print("Num val point cloud labels:", len(val_label_cloud))

        train_dataset = generate_dataset(train_point_clouds, train_label_cloud, BATCH_SIZE)
        val_dataset   = generate_dataset(val_point_clouds, val_label_cloud, BATCH_SIZE , is_training=False)
        
    elif mode == 'test':

        val_point_clouds  = point_clouds[:]
        val_label_cloud   = point_cloud_labels[:]
        val_dataset       = generate_dataset(val_point_clouds, val_label_cloud, BATCH_SIZE , is_training=False)
        #val_dataset = np.copy(train_dataset)



    """
    ## Instantiate the model
    """
    
    if mode == 'train':
        x, y = next(iter(train_dataset))

        num_points = x.shape[1]
        num_classes = y.shape[-1]
        print("MODEL NUM POINTS:", num_points)
        segmentation_model = get_shape_segmentation_model(num_points, num_classes)
        #segmentation_model.summary()
        """
        ## Training

        For the training the authors recommend using a learning rate schedule that decays the
        initial learning rate by half every 20 epochs. In this example, we resort to 15 epochs.
        """

        training_step_size = total_training_examples // BATCH_SIZE
        total_training_steps = training_step_size * EPOCHS

        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[training_step_size * 15, training_step_size * 15],
            values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
        )

        steps = tf.range(total_training_steps, dtype=tf.int32)
        lrs = [lr_schedule(step) for step in steps]
        
        print(f"Total training steps: {total_training_steps}.")
        segmentation_model, history = run_experiment(EPOCHS, train_dataset, val_dataset, lr_schedule, num_points, num_classes)
        w_file = str(np.copy(pc_file)).replace("atria_cp", "weigths")
        segmentation_model.save_weights(w_file[:-4]+".hf5")

    elif mode == 'test':
        x, y = next(iter(val_dataset))

        num_points = x.shape[1]
        num_classes = y.shape[-1]
        print("MODEL NUM POINTS:", num_points)
        segmentation_model = get_shape_segmentation_model(num_points, num_classes)
        #segmentation_model.summary()

        segmentation_model.load_weights(weigths_file)
         
    """
    ## Visualize the training landscape
    """

    #plot_result("loss")
    #plot_result("accuracy")


    """
    ## Inference
    """
    for i in range(len(val_dataset)):
      validation_batch = next(iter(val_dataset))
    
      val_predictions = segmentation_model.predict(validation_batch[0])
      print(f"Validation prediction shape: {val_predictions.shape}")
      
      # Plotting with ground-truth.
      visualize_single_point_cloud(validation_batch[0], validation_batch[1], 0, LABELS)
      # Plotting with predicted labels.
      visualize_single_point_cloud(validation_batch[0], val_predictions, 0, LABELS)

      from sklearn.metrics import classification_report
      #y_true = np.array(validation_batch[1][0,:,1]).astype(int)
      label_map = LABELS + ["none"]
      y_true = [label_map[np.argmax(label)] for label in validation_batch[1][0]]
      y_pred = [label_map[np.argmax(label)] for label in val_predictions[0]]

      print(classification_report(y_true, y_pred))
      


