"""
Title: SimSiam Training with TensorFlow Similarity and KerasCV
Author: [lukewood](https://lukewood.xyz), Ian Stenbit, Owen Vallis
Date created: 2022/09/21
Last modified: 2022/09/21
Description: Train a KerasCV model using unlabelled data with SimSiam.
"""

"""
## Overview

[TensorFlow similarity](https://github.com/tensorflow/similarity) makes it easy to train
KerasCV models on unlabelled corpuses of data using contrastive learning algorithms such
as SimCLR, SimSiam, and Barlow Twins.  In this guide, we will train a KerasCV model
using the SimSiam implementation from TensorFlow Similarity.

## Background

Self-supervised learning is an approach to pre-training models using unlabeled data. T
his approach drastically increases accuracy when you have very few labeled examples but
a lot of unlabelled data.
The key insight is that you can train a self-supervised model to learn data
representations by contrasting multiple augmented views of the same example.
These learned representations capture data invariants, e.g., object translation, color
jitter, noise, etc. Training a simple linear classifier on top of the frozen
representations is easier and requires fewer labels because the pre-trained model
already produces meaningful and generally useful features.

Overall, self-supervised pre-training learns representations which are more generic and
robust than other approaches to augmented training and pre-training.  An overview of
the general contrastive learning process is shown below:

![Contrastive overview](https://i.imgur.com/mzaEq3C.png)

In this tutorial, we will use the [SimSiam](https://arxiv.org/abs/2011.10566) algorithm
for contrastive learning.  As of 2022, SimSiam is the state of the art algorithm for
contrastive learning; allowing for unprecedented scores on CIFAR-100 and other datasets.

To get started, we will sort out some imports.
"""

import gc
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import tensorflow_similarity as tfsim  # main package
import tensorflow as tf

import tensorflow_datasets as tfds
tfsim.utils.tf_cap_memory()  # Avoid GPU memory blow up
tfds.disable_progress_bar()

"""
## Data Loading

Next, we will load the STL-10 dataset.  STL-10 is a dataset consisting of 100k unlabelled
images, 5k labelled training images, and 10k labelled test images.  Due to this distribution,
STL-10 is commonly used as a benchmark for contrastive learning models.

First lets load our unlabelled data
"""
unlabelled_images = tfds.load("stl10", split="unlabelled")
unlabelled_images = unlabelled_images.map(
    lambda entry: entry["image"], num_parallel_calls=tf.data.AUTOTUNE
)
unlabelled_images = unlabelled_images.shuffle(
    buffer_size = 8*BATCH_SIZE, reshuffle_each_iteration=True
)
unlabelled_images = unlabelled_images.batch(BATCH_SIZE)

"""
Next, we need to prepare some labelled samples.
This is done so that TensorFlow similarity can probe the learned embedding to ensure
that the model is learning appropriately.
"""

train_labelled_images, ds_info = tfds.load("stl10", split="train", as_supervised=True, batch_size=-1,)
test_labelled_images = tfds.load("stl10", split="test", as_supervised=True, batch_size=-1,)

# Compute the indicies for query, index, val, and train splits
query_idxs, index_idxs, val_idxs, train_idxs = [], [], [], []
for cid in range(ds_info.features["label"].num_classes):
    idxs = tf.random.shuffle(tf.where(y_raw_train == cid))
    idxs = tf.reshape(idxs, (-1,))
    query_idxs.extend(idxs[:200])  # 200 query examples per class
    index_idxs.extend(idxs[200:400])  # 200 index examples per class
    val_idxs.extend(idxs[400:500])  # 100 validation examples per class
    train_idxs.extend(idxs[500:])  # The remaining are used for training

random.shuffle(query_idxs)
random.shuffle(index_idxs)
random.shuffle(val_idxs)
random.shuffle(train_idxs)

def create_split(idxs: list) -> tuple:
    x, y = [], []
    for idx in idxs:
        x.append(x_raw_train[int(idx)])
        y.append(y_raw_train[int(idx)])
    return tf.convert_to_tensor(np.array(x)), tf.convert_to_tensor(np.array(y))

x_query, y_query = create_split(query_idxs)
x_index, y_index = create_split(index_idxs)
x_val, y_val = create_split(val_idxs)
x_train, y_train = create_split(train_idxs)

"""
## Augmentations
Self-supervised networks require at least two augmented "views" of each example. This can be created using a DataSet and an augmentation function. The DataSet treats each example in the batch as its own class and then the augment function produces two separate views for each example.

This means the resulting batch will yield tuples containing the two views, i.e.,
Tuple[(BATCH_SIZE, 32, 32, 3), (BATCH_SIZE, 32, 32, 3)].

TensorFlow Similarity provides several random augmentation functions, and here we combine augmenters from the simCLR module to replicate the augmentations used in simsiam.
"""

"""
Now that we have all of our datasets produced, we can move on to constructing the actual
models.  First, lets define some hyper-parameters that are typical for SimSiam:
"""
BATCH_SIZE = 512
PRE_TRAIN_EPOCHS = 800
PRE_TRAIN_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4
INIT_LR = 3e-2 * int(BATCH_SIZE / 256)
WARMUP_LR = 0.0
WARMUP_STEPS = 0
DIM = 2048
