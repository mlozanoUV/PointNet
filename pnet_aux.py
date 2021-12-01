import os
import nrrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from skimage.exposure import rescale_intensity, equalize_hist
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage import measure

from skimage.exposure import rescale_intensity
from skimage import filters
from skimage.feature import canny
from scipy import ndimage as ndi

from pnet_vis import mcubes
from skimage.segmentation import chan_vese,circle_level_set 

def show_bbox_scene(img, contours):
    plt.imshow(img)
    seg = chan_vese(img,mu=0.16)
    #seg = circle_level_set(img)
    plt.imshow(seg)    
    for c in contours:
        plt.scatter(c[:,0], c[:,1], c='r')
        
    plt.show()
    return
def adapt_bbox(bbox, w, h):
    # ajusto la bbox a tam fijo: w, h
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]
    if cw < w: 
        offset = (w - cw)//2
        bbox[0]-= offset
        bbox[2]+= offset
    if ch < h: 
        offset = (h - ch)//2
        bbox[1]-= offset
        bbox[3]+= offset
    
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]
    if cw < w: bbox[2]+=1
    if ch < h: bbox[3]+=1
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]

    return bbox
    
def get_bbox_from_mask(mask, offset=3):
    pixels  = np.argwhere(mask > 0)
    bbox = []
    if len(pixels) > 0:
        bbox  = [pixels.T[0].min()-offset, pixels.T[1].min()-offset, pixels.T[0].max()+offset, pixels.T[1].max()+offset]
    return bbox

def get_contours_from_mask(mask, img=[],offset=3):

    pixels  = np.argwhere(mask > 0)
    lpx = []  

    if len(pixels) > 0:
        bbox    = [pixels.T[0].min()-offset, pixels.T[1].min()-offset, pixels.T[0].max()+offset, pixels.T[1].max()+offset]
        amask   = mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        contours_gt = measure.find_contours(amask, fully_connected='high')

        if len(img)>0:
            aimg   = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            #show_bbox_scene(aimg.T, contours_gt)

        for contour in contours_gt:
            for px in contour:
                lpx.append(px)

    return lpx

PTOS3D_RESO = 336512
def convert(nfiles, npoints):
  Lr, Lm, npoints, Ln = load_nrrd_data("../RL-AtriaSeg/Training Set/", nfiles, npoints)
  pointsc, masksc = np.array(Lr), np.array(Lm)
  fname = "atria_cp."+str(nfiles)+"."+str(npoints)+".npy"
  mname = "atria_mask_cp."+str(nfiles)+"."+str(npoints)+".npy"
  np.save(fname, pointsc)
  np.save(mname, masksc)
  np.savetxt("atria_cp."+str(nfiles)+"."+str(npoints)+".ids.txt", np.array(Ln),fmt="%s")
  print("Saved: ", fname, mname, pointsc.shape, masksc.shape)

def get_AtriaContourPoints(reso_, mask_, npoints_per_image=-1, show_images = False):
   
    ax = []
    if show_images:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    
    Lxyz, Llabels = [],[]
    zmax = reso_.shape[2]
    vol = np.zeros([mask_.shape[0],mask_.shape[1],mask_.shape[2]]).astype(int)

    for z in tqdm( range(zmax)):
        img, mask = reso_[:,:,z],mask_[:,:,z]
        
        # Seleccion de los pixels
        pixels = get_contours_from_mask(mask, img)
        
        for i, p in enumerate(pixels):
            x, y = np.copy(int(p[0])), np.copy(int(p[1]))
            Lxyz.append( np.array([x,y,z]) )
            Llabels.append( 1  )
            vol[x,y,z] = 1
                
    print("Nptos (MRI): ", len(Lxyz))
    return np.array(Lxyz), np.array(Llabels), vol


def load_nrrd_data(training_path, maxFiles, npoints):
    Lmasks, Lresos, Lnames = [],[], []
    all_points = 0
    nfiles = 0

    #tr_ids = open("atria_cp.80.128.ids.txt").read().split('\n')
    for d in os.listdir(training_path):
        if len(d) > 10 and nfiles < maxFiles:# and (d in tr_ids) == False: 
            _mask, mask_header = nrrd.read(training_path+d+'/laendo.nrrd')
            _reso, reso_header = nrrd.read(training_path+d+'/lgemri.nrrd')
            
            #reso, mask = sub_sample(_reso, _mask, 128)
            reso, mask, vol = get_AtriaContourPoints(_reso, _mask)
            print(reso.shape, mask.shape, npoints)
            id_points = np.random.randint(0, reso.shape[0], npoints)
            reso = reso[id_points]
            mask = mask[id_points]

            
            #visualize_data(reso, mask)
            Lresos.append(reso)
            Lmasks.append(mask)
            Lnames.append(d)
            all_points+= len(reso)
            nfiles+=1 
            print(d, all_points)

            mcubes(_mask)

           
    return Lresos,Lmasks, reso.shape[0], Lnames 


def load_data_(point_cloud_batch, label_cloud_batch):
    print("load_data_: ", point_cloud_batch.shape, label_cloud_batch.shape)
    NUM_SAMPLE_POINTS = point_cloud_batch.shape[0]
    point_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    label_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    return point_cloud_batch, label_cloud_batch


def generate_dataset(points, labels, BATCH_SIZE, is_training=True):
    points = np.array(points)
    labels = np.array(labels)
    print("gs: ", points.shape, labels.shape)
    input(".......")
    dataset = tf.data.Dataset.from_tensor_slices((points, labels))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data_)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    #dataset = (
    #    dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    #    if is_training
    #    else dataset
    #)
    return dataset

    """
## PointNet model

The figure below depicts the internals of the PointNet model family:

![](https://i.imgur.com/qFLNw5L.png)

Given that PointNet is meant to consume an ***unordered set*** of coordinates as its input data,
its architecture needs to match the following characteristic properties
of point cloud data:

### Permutation invariance

Given the unstructured nature of point cloud data, a scan made up of `n` points has `n!`
permutations. The subsequent data processing must be invariant to the different
representations. In order to make PointNet invariant to input permutations, we use a
symmetric function (such as max-pooling) once the `n` input points are mapped to
higher-dimensional space. The result is a **global feature vector** that aims to capture
an aggregate signature of the `n` input points. The global feature vector is used alongside
local point features for segmentation.

![](https://i.imgur.com/0mrvvjb.png)

### Transformation invariance

Segmentation outputs should be unchanged if the object undergoes certain transformations,
such as translation or scaling. For a given input point cloud, we apply an appropriate
rigid or affine transformation to achieve pose normalization. Because each of the `n` input
points are represented as a vector and are mapped to the embedding spaces independently,
applying a geometric transformation simply amounts to matrix multiplying each point with
a transformation matrix. This is motivated by the concept of
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).

The operations comprising the T-Net are motivated by the higher-level architecture of
PointNet. MLPs (or fully-connected layers) are used to map the input points independently
and identically to a higher-dimensional space; max-pooling is used to encode a global
feature vector whose dimensionality is then reduced with fully-connected layers. The
input-dependent features at the final fully-connected layer are then combined with
globally trainable weights and biases, resulting in a 3-by-3 transformation matrix.

![](https://i.imgur.com/aEj3GYi.png)

### Point interactions

The interaction between neighboring points often carries useful information (i.e., a
single point should not be treated in isolation). Whereas classification need only make
use of global features, segmentation must be able to leverage local point features along
with global point features.


**Note**: The figures presented in this section have been taken from the
[original paper](https://arxiv.org/abs/1612.00593).
"""

"""
Now that we know the pieces that compose the PointNet model, we can implement the model.
We start by implementing the basic blocks i.e., the convolutional block and the multi-layer
perceptron block.
"""


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


"""
We implement a regularizer (taken from
[this example](https://keras.io/examples/vision/pointnet/#build-a-model))
to enforce orthogonality in the feature space. This is needed to ensure
that the magnitudes of the transformed features do not vary too much.
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config


"""
The next piece is the transformation network which we explained earlier.
"""


def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


"""
Finally, we piece the above blocks together and implement the segmentation model.
"""


def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)



def run_experiment(epochs, train_dataset, val_dataset, lr_schedule, num_points, num_classes):

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model, history


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def visualize_single_point_cloud(point_clouds, label_clouds, idx, LABELS):
    label_map = LABELS + ["none"]
    point_cloud = point_clouds[idx]
    label_cloud = label_clouds[idx]
    visualize_data(point_cloud, [label_map[np.argmax(label)] for label in label_cloud], LABELS)


