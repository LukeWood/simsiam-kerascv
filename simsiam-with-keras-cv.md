# SimSiam Training with TensorFlow Similarity and KerasCV

**Author:** [lukewood](https://lukewood.xyz), Ian Stenbit, Owen Vallis<br>
**Date created:** 2022/09/21<br>
**Last modified:** 2022/09/21<br>
**Description:** Train a KerasCV model using unlabelled data with SimSiam.

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


```python
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


import gc
import os
import random
import time
import tensorflow_addons as tfa
import keras_cv
from pathlib import Path
from tensorflow_similarity.layers import GeneralizedMeanPooling2D, MetricEmbedding
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tabulate import tabulate
import tensorflow_similarity as tfsim  # main package
import tensorflow as tf
from keras_cv import layers as cv_layers

import tensorflow_datasets as tfds

tfsim.utils.tf_cap_memory()  # Avoid GPU memory blow up
tfds.disable_progress_bar()

BATCH_SIZE = 512
PRE_TRAIN_EPOCHS = 50
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4
INIT_LR = 3e-2 * int(BATCH_SIZE / 256)
WARMUP_LR = 0.0
WARMUP_STEPS = 0
DIM = 2048
```

<div class="k-default-codeblock">
```
Your CPU supports instructions that this binary was not compiled to use: SSE3 SSE4.1 SSE4.2 AVX AVX2
For maximum performance, you can install NMSLIB from sources 
pip install --no-binary :all: nmslib

```
</div>
## Data Loading

Next, we will load the STL-10 dataset.  STL-10 is a dataset consisting of 100k unlabelled
images, 5k labelled training images, and 10k labelled test images.  Due to this distribution,
STL-10 is commonly used as a benchmark for contrastive learning models.

First lets load our unlabelled data


```python
train_ds = tfds.load("stl10", split="unlabelled")
train_ds = train_ds.map(
    lambda entry: entry["image"], num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.map(
    lambda image: tf.cast(image, tf.float32), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.shuffle(buffer_size=8 * BATCH_SIZE, reshuffle_each_iteration=True)
```

<div class="k-default-codeblock">
```
[1mDownloading and preparing dataset 2.46 GiB (download: 2.46 GiB, generated: 1.86 GiB, total: 4.32 GiB) to ~/tensorflow_datasets/stl10/1.0.0...[0m
[1mDataset stl10 downloaded and prepared to ~/tensorflow_datasets/stl10/1.0.0. Subsequent calls will reuse this data.[0m

```
</div>
Next, we need to prepare some labelled samples.
This is done so that TensorFlow similarity can probe the learned embedding to ensure
that the model is learning appropriately.


```python
(x_raw_train, y_raw_train), ds_info = tfds.load(
    "stl10", split="train", as_supervised=True, batch_size=-1, with_info=True
)
x_raw_train, y_raw_train = tf.cast(x_raw_train, tf.float32), tf.cast(
    y_raw_train, tf.float32
)
x_test, y_test = tfds.load(
    "stl10",
    split="test",
    as_supervised=True,
    batch_size=-1,
)
x_test, y_test = tf.cast(x_test, tf.float32), tf.cast(y_test, tf.float32)

# Compute the indicies for query, index, val, and train splits
query_idxs, index_idxs, val_idxs, train_idxs = [], [], [], []
for cid in range(ds_info.features["label"].num_classes):
    idxs = tf.random.shuffle(tf.where(y_raw_train == cid))
    idxs = tf.reshape(idxs, (-1,))
    query_idxs.extend(idxs[:100])  # 200 query examples per class
    index_idxs.extend(idxs[100:200])  # 200 index examples per class
    val_idxs.extend(idxs[200:300])  # 100 validation examples per class
    train_idxs.extend(idxs[300:])  # The remaining are used for training

random.shuffle(query_idxs)
random.shuffle(index_idxs)
random.shuffle(val_idxs)
random.shuffle(train_idxs)


def create_split(idxs: list) -> tuple:
    x, y = [], []
    for idx in idxs:
        x.append(x_raw_train[int(idx)])
        y.append(y_raw_train[int(idx)])
    return tf.convert_to_tensor(np.array(x), dtype=tf.float32), tf.convert_to_tensor(
        np.array(y), dtype=tf.int64
    )


x_query, y_query = create_split(query_idxs)
x_index, y_index = create_split(index_idxs)
x_val, y_val = create_split(val_idxs)
x_train, y_train = create_split(train_idxs)

PRE_TRAIN_STEPS_PER_EPOCH = tf.data.experimental.cardinality(train_ds) // BATCH_SIZE
PRE_TRAIN_STEPS_PER_EPOCH = int(PRE_TRAIN_STEPS_PER_EPOCH.numpy())

print(
    tabulate(
        [
            ["train", tf.data.experimental.cardinality(train_ds), None],
            ["val", x_val.shape, y_val.shape],
            ["query", x_query.shape, y_query.shape],
            ["index", x_index.shape, y_index.shape],
            ["test", x_test.shape, y_test.shape],
        ],
        headers=["Examples", "Labels"],
    )
)
```

<div class="k-default-codeblock">
```
       Examples           Labels
-----  -----------------  --------
train  100000
val    (1000, 96, 96, 3)  (1000,)
query  (1000, 96, 96, 3)  (1000,)
index  (1000, 96, 96, 3)  (1000,)
test   (8000, 96, 96, 3)  (8000,)

```
</div>
## Augmentations
Self-supervised networks require at least two augmented "views" of each example. This can be created using a DataSet and an augmentation function. The DataSet treats each example in the batch as its own class and then the augment function produces two separate views for each example.

This means the resulting batch will yield tuples containing the two views, i.e.,
Tuple[(BATCH_SIZE, 32, 32, 3), (BATCH_SIZE, 32, 32, 3)].

TensorFlow Similarity provides several random augmentation functions, and here we combine augmenters from the simCLR module to replicate the augmentations used in simsiam.

## Augmentation

Now that we have all of our datasets produced, we can move on to dataset augmentation.
Using KerasCV, it is trivial to construct an augmenter that performs as the one
described in the original SimSiam paper.  Lets do that below.


```python
target_size = (96, 96)
crop_area_factor = (0.08, 1)
aspect_ratio_factor = (3 / 4, 4 / 3)
grayscale_rate = 0.2
color_jitter_rate = 0.8
brightness_factor = 0.2
contrast_factor = 0.8
saturation_factor = (0.3, 0.7)
hue_factor = 0.2

augmenter = keras_cv.layers.Augmenter(
    [
        cv_layers.RandomFlip("horizontal"),
        cv_layers.RandomCropAndResize(
            target_size,
            crop_area_factor=crop_area_factor,
            aspect_ratio_factor=aspect_ratio_factor,
        ),
        cv_layers.MaybeApply(
            cv_layers.Grayscale(output_channels=3), rate=grayscale_rate
        ),
        cv_layers.MaybeApply(
            cv_layers.RandomColorJitter(
                value_range=(0, 255),
                brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                saturation_factor=saturation_factor,
                hue_factor=hue_factor,
            ),
            rate=color_jitter_rate,
        ),
    ],
)
```

Next, lets pass our images through this pipeline.
Note that KerasCV supports batched augmentation, so batching before
augmentation dramatically improves performance


```python

@tf.function()
def process(img):
    return augmenter(img), augmenter(img)


train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(1024)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(x_val)
val_ds = val_ds.repeat()
val_ds = val_ds.shuffle(1024)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print("train_ds", train_ds)
print("val_ds", val_ds)
```

<div class="k-default-codeblock">
```
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2

WARNING:tensorflow:Using a while_loop for converting CropAndResize

WARNING:tensorflow:Using a while_loop for converting CropAndResize

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip

WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting Bitcast

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2

WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2

WARNING:tensorflow:Using a while_loop for converting CropAndResize

WARNING:tensorflow:Using a while_loop for converting CropAndResize

train_ds <PrefetchDataset element_spec=(TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None))>
val_ds <PrefetchDataset element_spec=(TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None))>

```
</div>
Lets visualize our pairs using the `tfsim.visualization` utility package.


```python
display_imgs = next(train_ds.as_numpy_iterator())
max_pixel = np.max([display_imgs[0].max(), display_imgs[1].max()])
min_pixel = np.min([display_imgs[0].min(), display_imgs[1].min()])

tfsim.visualization.visualize_views(
    views=display_imgs,
    num_imgs=16,
    views_per_col=8,
    max_pixel_value=max_pixel,
    min_pixel_value=min_pixel,
)
```


    
![png](/simsiam-with-keras-cv/simsiam-with-keras-cv_13_0.png)
    


## Model Creation

Now that our data and augmentation pipeline is setup, we can move on to
constructing the contrastive learning pipeline.  First, lets produce a backbone.
For this task, we will use a KerasCV ResNet18 model as the backbone.


```python

def get_backbone(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = keras_cv.models.ResNet18(
        input_shape=input_shape,
        include_rescaling=True,
        include_top=False,
    )(x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    return tfsim.models.SimilarityModel(inputs, x)


backbone = get_backbone((96, 96, 3))
backbone.summary()
```

<div class="k-default-codeblock">
```
Model: "similarity_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 96, 96, 3)]       0         
                                                                 
 resnet18 (Functional)       (None, 3, 3, 512)         11186112  
                                                                 
 avg_pool (GlobalAveragePool  (None, 512)              0         
 ing2D)                                                          
                                                                 
=================================================================
Total params: 11,186,112
Trainable params: 11,176,512
Non-trainable params: 9,600
_________________________________________________________________

```
</div>
This MLP is common to all the self-supervised models and is typically a stack of 3
layers of the same size. However, SimSiam only uses 2 layers for the smaller CIFAR
images. Having too much capacity in the models can make it difficult for the loss to
stabilize and converge.

Additionally, the SimSiam paper found that disabling the center and scale parameters
can lead to a small boost in the final loss.

NOTE This is the model output that is returned by `ContrastiveModel.predict()` and
represents the distance based embedding. This embedding can be used for the KNN
lookups and matching classification metrics. However, when using the pre-train
model for downstream tasks, only the `ContrastiveModel.backbone` is used.


```python

def get_projector(input_dim, dim, activation="relu", num_layers: int = 3):
    inputs = tf.keras.layers.Input((input_dim,), name="projector_input")
    x = inputs

    for i in range(num_layers - 1):
        x = tf.keras.layers.Dense(
            dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name=f"projector_layer_{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1.001e-5, name=f"batch_normalization_{i}"
        )(x)
        x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_{i}")(
            x
        )
    x = tf.keras.layers.Dense(
        dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="projector_output",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1.001e-5,
        center=False,  # Page:5, Paragraph:2 of SimSiam paper
        scale=False,  # Page:5, Paragraph:2 of SimSiam paper
        name=f"batch_normalization_ouput",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    o = tfsim.layers.ActivationStdLoggingLayer(name="proj_std")(x)
    projector = tf.keras.Model(inputs, o, name="projector")
    return projector


projector = get_projector(input_dim=backbone.output.shape[-1], dim=DIM, num_layers=2)
projector.summary()

```

<div class="k-default-codeblock">
```
Model: "projector"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 projector_input (InputLayer  [(None, 512)]            0         
 )                                                               
                                                                 
 projector_layer_0 (Dense)   (None, 2048)              1048576   
                                                                 
 batch_normalization_0 (Batc  (None, 2048)             8192      
 hNormalization)                                                 
                                                                 
 relu_activation_0 (Activati  (None, 2048)             0         
 on)                                                             
                                                                 
 projector_output (Dense)    (None, 2048)              4194304   
                                                                 
 batch_normalization_ouput (  (None, 2048)             4096      
 BatchNormalization)                                             
                                                                 
 proj_std (ActivationStdLogg  (None, 2048)             0         
 ingLayer)                                                       
                                                                 
=================================================================
Total params: 5,255,168
Trainable params: 5,246,976
Non-trainable params: 8,192
_________________________________________________________________

```
</div>
Finally, we must construct the predictor.  The predictor is used in SimSiam, and is a
simple stack of two MLP layers, containing a bottleneck in the hidden layer.


```python

def get_predictor(input_dim, hidden_dim=512, activation="relu"):
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="predictor_input")
    x = inputs

    x = tf.keras.layers.Dense(
        hidden_dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_layer_0",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1.001e-5, name="batch_normalization_0"
    )(x)
    x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_0")(x)

    x = tf.keras.layers.Dense(
        input_dim,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_output",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    o = tfsim.layers.ActivationStdLoggingLayer(name="pred_std")(x)
    predictor = tf.keras.Model(inputs, o, name="predictor")
    return predictor


predictor = get_predictor(input_dim=DIM, hidden_dim=512)
predictor.summary()

```

<div class="k-default-codeblock">
```
Model: "predictor"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 predictor_input (InputLayer  [(None, 2048)]           0         
 )                                                               
                                                                 
 predictor_layer_0 (Dense)   (None, 512)               1048576   
                                                                 
 batch_normalization_0 (Batc  (None, 512)              2048      
 hNormalization)                                                 
                                                                 
 relu_activation_0 (Activati  (None, 512)              0         
 on)                                                             
                                                                 
 predictor_output (Dense)    (None, 2048)              1050624   
                                                                 
 pred_std (ActivationStdLogg  (None, 2048)             0         
 ingLayer)                                                       
                                                                 
=================================================================
Total params: 2,101,248
Trainable params: 2,100,224
Non-trainable params: 1,024
_________________________________________________________________

```
</div>
## Training

First, we need to initialize our training model, loss, and optimizer.


```python
loss = tfsim.losses.SimSiamLoss(projection_type="cosine_distance", name="simsiam")

contrastive_model = tfsim.models.ContrastiveModel(
    backbone=backbone,
    projector=projector,
    predictor=predictor,  # NOTE: simiam requires predictor model.
    algorithm="simsiam",
    name="simsiam",
)
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
)
wd_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=WEIGHT_DECAY,
    decay_steps=PRE_TRAIN_EPOCHS * PRE_TRAIN_STEPS_PER_EPOCH,
)
optimizer = tfa.optimizers.SGDW(
    learning_rate=lr_decayed_fn, weight_decay=wd_decayed_fn, momentum=0.9
)
```

Next we can compile the model the same way you compile any other Keras model.


```python
contrastive_model.compile(
    optimizer=optimizer,
    loss=loss,
)
```

We track the training using several callbacks.

* **EvalCallback** creates an index at the end of each epoch and provides a proxy for the nearest neighbor matching classification using `binary_accuracy`.
* **TensordBoard** and **ModelCheckpoint** are provided for tracking the training progress.


```python
DATA_PATH = Path("./")
log_dir = DATA_PATH / "models" / "logs" / f"{loss.name}_{time.time()}"
chkpt_dir = DATA_PATH / "models" / "checkpoints" / f"{loss.name}_{time.time()}"

callbacks = [
    tfsim.callbacks.EvalCallback(
        tf.cast(x_query, tf.float32),
        y_query,
        tf.cast(x_index, tf.float32),
        y_index,
        metrics=["binary_accuracy"],
        k=1,
        tb_logdir=log_dir,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq=100,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    ),
]
```

<div class="k-default-codeblock">
```
TensorBoard logging enable in models/logs/simsiam_1664055128.3967984/index

```
</div>
All that is left to do is run fit()!


```python
print(train_ds)
print(val_ds)
history = contrastive_model.fit(
    train_ds,
    epochs=PRE_TRAIN_EPOCHS,
    steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
    callbacks=callbacks,
)

```

<div class="k-default-codeblock">
```
<PrefetchDataset element_spec=(TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None))>
<PrefetchDataset element_spec=(TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32, name=None))>
Epoch 1/50
  6/195 [..............................] - ETA: 1:59 - loss: 0.9972 - proj_std: 0.0217 - pred_std: 0.0179WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.2253s vs `on_train_batch_end` time: 0.3624s). Check your callbacks.

WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.2253s vs `on_train_batch_end` time: 0.3624s). Check your callbacks.

195/195 [==============================] - ETA: 0s - loss: 0.2814 - proj_std: 0.0116 - pred_std: 0.0071binary_accuracy: 0.2280
195/195 [==============================] - 223s 1s/step - loss: 0.2814 - proj_std: 0.0116 - pred_std: 0.0071 - val_loss: 0.1007 - val_proj_std: 0.0082 - val_pred_std: 0.0036 - binary_accuracy: 0.2280
Epoch 2/50
195/195 [==============================] - ETA: 0s - loss: 0.1482 - proj_std: 0.0101 - pred_std: 0.0048binary_accuracy: 0.2020
195/195 [==============================] - 207s 1s/step - loss: 0.1482 - proj_std: 0.0101 - pred_std: 0.0048 - val_loss: 0.1699 - val_proj_std: 0.0114 - val_pred_std: 0.0065 - binary_accuracy: 0.2020
Epoch 3/50
195/195 [==============================] - ETA: 0s - loss: 0.2825 - proj_std: 0.0153 - pred_std: 0.0108binary_accuracy: 0.2540
195/195 [==============================] - 203s 1s/step - loss: 0.2825 - proj_std: 0.0153 - pred_std: 0.0108 - val_loss: 0.1406 - val_proj_std: 0.0092 - val_pred_std: 0.0072 - binary_accuracy: 0.2540
Epoch 4/50
195/195 [==============================] - ETA: 0s - loss: 0.2580 - proj_std: 0.0172 - pred_std: 0.0153binary_accuracy: 0.2730
195/195 [==============================] - 200s 1s/step - loss: 0.2580 - proj_std: 0.0172 - pred_std: 0.0153 - val_loss: 0.1583 - val_proj_std: 0.0117 - val_pred_std: 0.0096 - binary_accuracy: 0.2730
Epoch 5/50
195/195 [==============================] - ETA: 0s - loss: 0.2254 - proj_std: 0.0179 - pred_std: 0.0164binary_accuracy: 0.2620
195/195 [==============================] - 197s 1s/step - loss: 0.2254 - proj_std: 0.0179 - pred_std: 0.0164 - val_loss: 0.1387 - val_proj_std: 0.0133 - val_pred_std: 0.0117 - binary_accuracy: 0.2620
Epoch 6/50
195/195 [==============================] - ETA: 0s - loss: 0.2069 - proj_std: 0.0182 - pred_std: 0.0170binary_accuracy: 0.2730
195/195 [==============================] - 197s 1s/step - loss: 0.2069 - proj_std: 0.0182 - pred_std: 0.0170 - val_loss: 0.2540 - val_proj_std: 0.0194 - val_pred_std: 0.0184 - binary_accuracy: 0.2730
Epoch 7/50
195/195 [==============================] - ETA: 0s - loss: 0.2089 - proj_std: 0.0185 - pred_std: 0.0173binary_accuracy: 0.2940
195/195 [==============================] - 199s 1s/step - loss: 0.2089 - proj_std: 0.0185 - pred_std: 0.0173 - val_loss: 0.1323 - val_proj_std: 0.0134 - val_pred_std: 0.0113 - binary_accuracy: 0.2940
Epoch 8/50
195/195 [==============================] - ETA: 0s - loss: 0.2149 - proj_std: 0.0187 - pred_std: 0.0175binary_accuracy: 0.2870
195/195 [==============================] - 198s 1s/step - loss: 0.2149 - proj_std: 0.0187 - pred_std: 0.0175 - val_loss: 0.1411 - val_proj_std: 0.0113 - val_pred_std: 0.0107 - binary_accuracy: 0.2870
Epoch 9/50
195/195 [==============================] - ETA: 0s - loss: 0.2061 - proj_std: 0.0189 - pred_std: 0.0178binary_accuracy: 0.2860
195/195 [==============================] - 198s 1s/step - loss: 0.2061 - proj_std: 0.0189 - pred_std: 0.0178 - val_loss: 0.1854 - val_proj_std: 0.0173 - val_pred_std: 0.0157 - binary_accuracy: 0.2860
Epoch 10/50
195/195 [==============================] - ETA: 0s - loss: 0.1949 - proj_std: 0.0188 - pred_std: 0.0176binary_accuracy: 0.2990
195/195 [==============================] - 198s 1s/step - loss: 0.1949 - proj_std: 0.0188 - pred_std: 0.0176 - val_loss: 0.1137 - val_proj_std: 0.0135 - val_pred_std: 0.0120 - binary_accuracy: 0.2990
Epoch 11/50
195/195 [==============================] - ETA: 0s - loss: 0.2144 - proj_std: 0.0191 - pred_std: 0.0179binary_accuracy: 0.3080
195/195 [==============================] - 199s 1s/step - loss: 0.2144 - proj_std: 0.0191 - pred_std: 0.0179 - val_loss: 0.2417 - val_proj_std: 0.0171 - val_pred_std: 0.0166 - binary_accuracy: 0.3080
Epoch 12/50
195/195 [==============================] - ETA: 0s - loss: 0.2217 - proj_std: 0.0198 - pred_std: 0.0188binary_accuracy: 0.3140
195/195 [==============================] - 198s 1s/step - loss: 0.2217 - proj_std: 0.0198 - pred_std: 0.0188 - val_loss: 0.1854 - val_proj_std: 0.0177 - val_pred_std: 0.0167 - binary_accuracy: 0.3140
Epoch 13/50
195/195 [==============================] - ETA: 0s - loss: 0.2303 - proj_std: 0.0201 - pred_std: 0.0193binary_accuracy: 0.3030
195/195 [==============================] - 199s 1s/step - loss: 0.2303 - proj_std: 0.0201 - pred_std: 0.0193 - val_loss: 0.2599 - val_proj_std: 0.0195 - val_pred_std: 0.0191 - binary_accuracy: 0.3030
Epoch 14/50
195/195 [==============================] - ETA: 0s - loss: 0.2177 - proj_std: 0.0203 - pred_std: 0.0197binary_accuracy: 0.3400
195/195 [==============================] - 197s 1s/step - loss: 0.2177 - proj_std: 0.0203 - pred_std: 0.0197 - val_loss: 0.2281 - val_proj_std: 0.0186 - val_pred_std: 0.0182 - binary_accuracy: 0.3400
Epoch 15/50
195/195 [==============================] - ETA: 0s - loss: 0.2134 - proj_std: 0.0202 - pred_std: 0.0196binary_accuracy: 0.3060
195/195 [==============================] - 200s 1s/step - loss: 0.2134 - proj_std: 0.0202 - pred_std: 0.0196 - val_loss: 0.1989 - val_proj_std: 0.0160 - val_pred_std: 0.0150 - binary_accuracy: 0.3060
Epoch 16/50
195/195 [==============================] - ETA: 0s - loss: 0.2101 - proj_std: 0.0206 - pred_std: 0.0201binary_accuracy: 0.3360
195/195 [==============================] - 198s 1s/step - loss: 0.2101 - proj_std: 0.0206 - pred_std: 0.0201 - val_loss: 0.1945 - val_proj_std: 0.0195 - val_pred_std: 0.0190 - binary_accuracy: 0.3360
Epoch 17/50
195/195 [==============================] - ETA: 0s - loss: 0.2107 - proj_std: 0.0203 - pred_std: 0.0197binary_accuracy: 0.3210
195/195 [==============================] - 198s 1s/step - loss: 0.2107 - proj_std: 0.0203 - pred_std: 0.0197 - val_loss: 0.2360 - val_proj_std: 0.0187 - val_pred_std: 0.0181 - binary_accuracy: 0.3210
Epoch 18/50
195/195 [==============================] - ETA: 0s - loss: 0.1933 - proj_std: 0.0197 - pred_std: 0.0191binary_accuracy: 0.3330
195/195 [==============================] - 200s 1s/step - loss: 0.1933 - proj_std: 0.0197 - pred_std: 0.0191 - val_loss: 0.3360 - val_proj_std: 0.0194 - val_pred_std: 0.0197 - binary_accuracy: 0.3330
Epoch 19/50
195/195 [==============================] - ETA: 0s - loss: 0.1944 - proj_std: 0.0198 - pred_std: 0.0193binary_accuracy: 0.3210
195/195 [==============================] - 198s 1s/step - loss: 0.1944 - proj_std: 0.0198 - pred_std: 0.0193 - val_loss: 0.2087 - val_proj_std: 0.0195 - val_pred_std: 0.0193 - binary_accuracy: 0.3210
Epoch 20/50
195/195 [==============================] - ETA: 0s - loss: 0.1904 - proj_std: 0.0196 - pred_std: 0.0191binary_accuracy: 0.3490
195/195 [==============================] - 199s 1s/step - loss: 0.1904 - proj_std: 0.0196 - pred_std: 0.0191 - val_loss: 0.1970 - val_proj_std: 0.0182 - val_pred_std: 0.0177 - binary_accuracy: 0.3490
Epoch 21/50
195/195 [==============================] - ETA: 0s - loss: 0.1834 - proj_std: 0.0190 - pred_std: 0.0185binary_accuracy: 0.3560
195/195 [==============================] - 198s 1s/step - loss: 0.1834 - proj_std: 0.0190 - pred_std: 0.0185 - val_loss: 0.2201 - val_proj_std: 0.0164 - val_pred_std: 0.0162 - binary_accuracy: 0.3560
Epoch 22/50
195/195 [==============================] - ETA: 0s - loss: 0.1596 - proj_std: 0.0186 - pred_std: 0.0181binary_accuracy: 0.3860
195/195 [==============================] - 198s 1s/step - loss: 0.1596 - proj_std: 0.0186 - pred_std: 0.0181 - val_loss: 0.1623 - val_proj_std: 0.0193 - val_pred_std: 0.0191 - binary_accuracy: 0.3860
Epoch 23/50
195/195 [==============================] - ETA: 0s - loss: 0.1560 - proj_std: 0.0188 - pred_std: 0.0184binary_accuracy: 0.3760
195/195 [==============================] - 198s 1s/step - loss: 0.1560 - proj_std: 0.0188 - pred_std: 0.0184 - val_loss: 0.1871 - val_proj_std: 0.0188 - val_pred_std: 0.0187 - binary_accuracy: 0.3760
Epoch 24/50
195/195 [==============================] - ETA: 0s - loss: 0.1522 - proj_std: 0.0187 - pred_std: 0.0182binary_accuracy: 0.3880
195/195 [==============================] - 198s 1s/step - loss: 0.1522 - proj_std: 0.0187 - pred_std: 0.0182 - val_loss: 0.1827 - val_proj_std: 0.0154 - val_pred_std: 0.0146 - binary_accuracy: 0.3880
Epoch 25/50
195/195 [==============================] - ETA: 0s - loss: 0.1435 - proj_std: 0.0183 - pred_std: 0.0178binary_accuracy: 0.4030
195/195 [==============================] - 197s 1s/step - loss: 0.1435 - proj_std: 0.0183 - pred_std: 0.0178 - val_loss: 0.1657 - val_proj_std: 0.0187 - val_pred_std: 0.0184 - binary_accuracy: 0.4030
Epoch 26/50
195/195 [==============================] - ETA: 0s - loss: 0.1334 - proj_std: 0.0179 - pred_std: 0.0174binary_accuracy: 0.3600
195/195 [==============================] - 199s 1s/step - loss: 0.1334 - proj_std: 0.0179 - pred_std: 0.0174 - val_loss: 0.1482 - val_proj_std: 0.0175 - val_pred_std: 0.0171 - binary_accuracy: 0.3600
Epoch 27/50
195/195 [==============================] - ETA: 0s - loss: 0.1284 - proj_std: 0.0185 - pred_std: 0.0180binary_accuracy: 0.4120
195/195 [==============================] - 197s 1s/step - loss: 0.1284 - proj_std: 0.0185 - pred_std: 0.0180 - val_loss: 0.2317 - val_proj_std: 0.0188 - val_pred_std: 0.0187 - binary_accuracy: 0.4120
Epoch 28/50
195/195 [==============================] - ETA: 0s - loss: 0.1213 - proj_std: 0.0183 - pred_std: 0.0179binary_accuracy: 0.3860
195/195 [==============================] - 199s 1s/step - loss: 0.1213 - proj_std: 0.0183 - pred_std: 0.0179 - val_loss: 0.1890 - val_proj_std: 0.0161 - val_pred_std: 0.0155 - binary_accuracy: 0.3860
Epoch 29/50
195/195 [==============================] - ETA: 0s - loss: 0.1231 - proj_std: 0.0183 - pred_std: 0.0180binary_accuracy: 0.4280
195/195 [==============================] - 197s 1s/step - loss: 0.1231 - proj_std: 0.0183 - pred_std: 0.0180 - val_loss: 0.1430 - val_proj_std: 0.0156 - val_pred_std: 0.0151 - binary_accuracy: 0.4280
Epoch 30/50
195/195 [==============================] - ETA: 0s - loss: 0.1235 - proj_std: 0.0184 - pred_std: 0.0180binary_accuracy: 0.4130
195/195 [==============================] - 197s 1s/step - loss: 0.1235 - proj_std: 0.0184 - pred_std: 0.0180 - val_loss: 0.1440 - val_proj_std: 0.0159 - val_pred_std: 0.0154 - binary_accuracy: 0.4130
Epoch 31/50
195/195 [==============================] - ETA: 0s - loss: 0.1153 - proj_std: 0.0182 - pred_std: 0.0179binary_accuracy: 0.4570
195/195 [==============================] - 199s 1s/step - loss: 0.1153 - proj_std: 0.0182 - pred_std: 0.0179 - val_loss: 0.1430 - val_proj_std: 0.0180 - val_pred_std: 0.0179 - binary_accuracy: 0.4570
Epoch 32/50
195/195 [==============================] - ETA: 0s - loss: 0.1183 - proj_std: 0.0181 - pred_std: 0.0178binary_accuracy: 0.4380
195/195 [==============================] - 197s 1s/step - loss: 0.1183 - proj_std: 0.0181 - pred_std: 0.0178 - val_loss: 0.1301 - val_proj_std: 0.0189 - val_pred_std: 0.0187 - binary_accuracy: 0.4380
Epoch 33/50
195/195 [==============================] - ETA: 0s - loss: 0.1134 - proj_std: 0.0179 - pred_std: 0.0176binary_accuracy: 0.4480
195/195 [==============================] - 198s 1s/step - loss: 0.1134 - proj_std: 0.0179 - pred_std: 0.0176 - val_loss: 0.1426 - val_proj_std: 0.0193 - val_pred_std: 0.0192 - binary_accuracy: 0.4480
Epoch 34/50
195/195 [==============================] - ETA: 0s - loss: 0.1172 - proj_std: 0.0179 - pred_std: 0.0176binary_accuracy: 0.4430
195/195 [==============================] - 197s 1s/step - loss: 0.1172 - proj_std: 0.0179 - pred_std: 0.0176 - val_loss: 0.1464 - val_proj_std: 0.0164 - val_pred_std: 0.0162 - binary_accuracy: 0.4430
Epoch 35/50
195/195 [==============================] - ETA: 0s - loss: 0.1142 - proj_std: 0.0176 - pred_std: 0.0174binary_accuracy: 0.4610
195/195 [==============================] - 197s 1s/step - loss: 0.1142 - proj_std: 0.0176 - pred_std: 0.0174 - val_loss: 0.1538 - val_proj_std: 0.0191 - val_pred_std: 0.0190 - binary_accuracy: 0.4610
Epoch 36/50
195/195 [==============================] - ETA: 0s - loss: 0.1146 - proj_std: 0.0177 - pred_std: 0.0175binary_accuracy: 0.4590
195/195 [==============================] - 197s 1s/step - loss: 0.1146 - proj_std: 0.0177 - pred_std: 0.0175 - val_loss: 0.1504 - val_proj_std: 0.0179 - val_pred_std: 0.0177 - binary_accuracy: 0.4590
Epoch 37/50
195/195 [==============================] - ETA: 0s - loss: 0.1217 - proj_std: 0.0176 - pred_std: 0.0174binary_accuracy: 0.4720
195/195 [==============================] - 198s 1s/step - loss: 0.1217 - proj_std: 0.0176 - pred_std: 0.0174 - val_loss: 0.1516 - val_proj_std: 0.0189 - val_pred_std: 0.0189 - binary_accuracy: 0.4720
Epoch 38/50
195/195 [==============================] - ETA: 0s - loss: 0.1119 - proj_std: 0.0172 - pred_std: 0.0169binary_accuracy: 0.4470
195/195 [==============================] - 197s 1s/step - loss: 0.1119 - proj_std: 0.0172 - pred_std: 0.0169 - val_loss: 0.1190 - val_proj_std: 0.0194 - val_pred_std: 0.0193 - binary_accuracy: 0.4470
Epoch 39/50
195/195 [==============================] - ETA: 0s - loss: 0.1015 - proj_std: 0.0174 - pred_std: 0.0171binary_accuracy: 0.4750
195/195 [==============================] - 196s 1s/step - loss: 0.1015 - proj_std: 0.0174 - pred_std: 0.0171 - val_loss: 0.1357 - val_proj_std: 0.0160 - val_pred_std: 0.0160 - binary_accuracy: 0.4750
Epoch 40/50
195/195 [==============================] - ETA: 0s - loss: 0.1070 - proj_std: 0.0171 - pred_std: 0.0169binary_accuracy: 0.4860
195/195 [==============================] - 199s 1s/step - loss: 0.1070 - proj_std: 0.0171 - pred_std: 0.0169 - val_loss: 0.1237 - val_proj_std: 0.0175 - val_pred_std: 0.0173 - binary_accuracy: 0.4860
Epoch 41/50
195/195 [==============================] - ETA: 0s - loss: 0.0985 - proj_std: 0.0170 - pred_std: 0.0168binary_accuracy: 0.4900
195/195 [==============================] - 197s 1s/step - loss: 0.0985 - proj_std: 0.0170 - pred_std: 0.0168 - val_loss: 0.1152 - val_proj_std: 0.0188 - val_pred_std: 0.0187 - binary_accuracy: 0.4900
Epoch 42/50
195/195 [==============================] - ETA: 0s - loss: 0.1005 - proj_std: 0.0168 - pred_std: 0.0167binary_accuracy: 0.5170
195/195 [==============================] - 197s 1s/step - loss: 0.1005 - proj_std: 0.0168 - pred_std: 0.0167 - val_loss: 0.1321 - val_proj_std: 0.0163 - val_pred_std: 0.0162 - binary_accuracy: 0.5170
Epoch 43/50
195/195 [==============================] - ETA: 0s - loss: 0.1028 - proj_std: 0.0168 - pred_std: 0.0167binary_accuracy: 0.4960
195/195 [==============================] - 197s 1s/step - loss: 0.1028 - proj_std: 0.0168 - pred_std: 0.0167 - val_loss: 0.1150 - val_proj_std: 0.0170 - val_pred_std: 0.0170 - binary_accuracy: 0.4960
Epoch 44/50
195/195 [==============================] - ETA: 0s - loss: 0.1012 - proj_std: 0.0167 - pred_std: 0.0166binary_accuracy: 0.5030
195/195 [==============================] - 197s 1s/step - loss: 0.1012 - proj_std: 0.0167 - pred_std: 0.0166 - val_loss: 0.1063 - val_proj_std: 0.0171 - val_pred_std: 0.0170 - binary_accuracy: 0.5030
Epoch 45/50
195/195 [==============================] - ETA: 0s - loss: 0.1006 - proj_std: 0.0165 - pred_std: 0.0165binary_accuracy: 0.4960
195/195 [==============================] - 198s 1s/step - loss: 0.1006 - proj_std: 0.0165 - pred_std: 0.0165 - val_loss: 0.1211 - val_proj_std: 0.0168 - val_pred_std: 0.0168 - binary_accuracy: 0.4960
Epoch 46/50
195/195 [==============================] - ETA: 0s - loss: 0.0950 - proj_std: 0.0166 - pred_std: 0.0165binary_accuracy: 0.5110
195/195 [==============================] - 198s 1s/step - loss: 0.0950 - proj_std: 0.0166 - pred_std: 0.0165 - val_loss: 0.1015 - val_proj_std: 0.0181 - val_pred_std: 0.0181 - binary_accuracy: 0.5110
Epoch 47/50
195/195 [==============================] - ETA: 0s - loss: 0.0930 - proj_std: 0.0165 - pred_std: 0.0164binary_accuracy: 0.5150
195/195 [==============================] - 197s 1s/step - loss: 0.0930 - proj_std: 0.0165 - pred_std: 0.0164 - val_loss: 0.1018 - val_proj_std: 0.0179 - val_pred_std: 0.0178 - binary_accuracy: 0.5150
Epoch 48/50
195/195 [==============================] - ETA: 0s - loss: 0.0939 - proj_std: 0.0165 - pred_std: 0.0164binary_accuracy: 0.5190
195/195 [==============================] - 200s 1s/step - loss: 0.0939 - proj_std: 0.0165 - pred_std: 0.0164 - val_loss: 0.0941 - val_proj_std: 0.0174 - val_pred_std: 0.0174 - binary_accuracy: 0.5190
Epoch 49/50
195/195 [==============================] - ETA: 0s - loss: 0.0929 - proj_std: 0.0165 - pred_std: 0.0164binary_accuracy: 0.5070
195/195 [==============================] - 196s 1s/step - loss: 0.0929 - proj_std: 0.0165 - pred_std: 0.0164 - val_loss: 0.0954 - val_proj_std: 0.0176 - val_pred_std: 0.0176 - binary_accuracy: 0.5070
Epoch 50/50
195/195 [==============================] - ETA: 0s - loss: 0.0917 - proj_std: 0.0164 - pred_std: 0.0164binary_accuracy: 0.5060
195/195 [==============================] - 198s 1s/step - loss: 0.0917 - proj_std: 0.0164 - pred_std: 0.0164 - val_loss: 0.0922 - val_proj_std: 0.0176 - val_pred_std: 0.0176 - binary_accuracy: 0.5060

```
</div>
## Plotting and Evaluation


```python
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history["loss"])
plt.grid()
plt.title(f"{loss.name} - loss")

plt.subplot(1, 3, 2)
plt.plot(history.history["proj_std"], label="proj")
if "pred_std" in history.history:
    plt.plot(history.history["pred_std"], label="pred")
plt.grid()
plt.title(f"{loss.name} - std metrics")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history["binary_accuracy"], label="acc")
plt.grid()
plt.title(f"{loss.name} - match metrics")
plt.legend()

plt.show()

```


    
![png](/simsiam-with-keras-cv/simsiam-with-keras-cv_29_0.png)
    


## Fine Tuning on the Labelled Data
TODO


```python
eval_augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomCropAndResize(
            (96, 96), crop_area_factor=(0.2, 1.0), aspect_ratio_factor=(1.0, 1.0)
        ),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
)

eval_train_ds = tf.data.Dataset.from_tensor_slices(
    (x_raw_train, tf.keras.utils.to_categorical(y_raw_train, 10))
)
eval_train_ds = eval_train_ds.repeat()
eval_train_ds = eval_train_ds.shuffle(1024)
eval_train_ds = eval_train_ds.map(lambda x, y: (eval_augmenter(x), y), tf.data.AUTOTUNE)
eval_train_ds = eval_train_ds.batch(BATCH_SIZE)
eval_train_ds = eval_train_ds.prefetch(tf.data.AUTOTUNE)

eval_val_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, tf.keras.utils.to_categorical(y_test, 10))
)
eval_val_ds = eval_val_ds.repeat()
eval_val_ds = eval_val_ds.shuffle(1024)
eval_val_ds = eval_val_ds.batch(BATCH_SIZE)
eval_val_ds = eval_val_ds.prefetch(tf.data.AUTOTUNE)
```

## Benchmark Against a Naive Model


```python
TEST_EPOCHS = 50
TEST_STEPS_PER_EPOCH = x_raw_train.shape[0] // BATCH_SIZE


def get_eval_model(img_size, backbone, total_steps, trainable=True, lr=1.8):
    backbone.trainable = trainable
    inputs = tf.keras.layers.Input((img_size, img_size, 3), name="eval_input")
    x = backbone(inputs, training=trainable)
    o = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, o)
    cosine_decayed_lr = tf.keras.experimental.CosineDecay(
        initial_learning_rate=lr, decay_steps=total_steps
    )
    opt = tf.keras.optimizers.SGD(cosine_decayed_lr, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
    return model


no_pt_eval_model = get_eval_model(
    img_size=96,
    backbone=get_backbone((96, 96, 3)),
    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
    trainable=True,
    lr=1e-3,
)
no_pt_history = no_pt_eval_model.fit(
    eval_train_ds,
    batch_size=BATCH_SIZE,
    epochs=TEST_EPOCHS,
    steps_per_epoch=TEST_STEPS_PER_EPOCH,
    validation_data=eval_val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
)
```

<div class="k-default-codeblock">
```
Epoch 1/50
9/9 [==============================] - 8s 752ms/step - loss: 2.7204 - acc: 0.1217 - val_loss: 2.2432 - val_acc: 0.1646
Epoch 2/50
9/9 [==============================] - 5s 628ms/step - loss: 2.1729 - acc: 0.2120 - val_loss: 2.1038 - val_acc: 0.2368
Epoch 3/50
9/9 [==============================] - 5s 638ms/step - loss: 1.9487 - acc: 0.2921 - val_loss: 1.8827 - val_acc: 0.2947
Epoch 4/50
9/9 [==============================] - 5s 643ms/step - loss: 1.7227 - acc: 0.3576 - val_loss: 1.7485 - val_acc: 0.3293
Epoch 5/50
9/9 [==============================] - 6s 644ms/step - loss: 1.5611 - acc: 0.4362 - val_loss: 1.6582 - val_acc: 0.3666
Epoch 6/50
9/9 [==============================] - 5s 636ms/step - loss: 1.4442 - acc: 0.4933 - val_loss: 1.6198 - val_acc: 0.3867
Epoch 7/50
9/9 [==============================] - 5s 629ms/step - loss: 1.3537 - acc: 0.5454 - val_loss: 1.5864 - val_acc: 0.4018
Epoch 8/50
9/9 [==============================] - 5s 624ms/step - loss: 1.2386 - acc: 0.6207 - val_loss: 1.5620 - val_acc: 0.4099
Epoch 9/50
9/9 [==============================] - 5s 617ms/step - loss: 1.1510 - acc: 0.6667 - val_loss: 1.5457 - val_acc: 0.4124
Epoch 10/50
9/9 [==============================] - 5s 613ms/step - loss: 1.0705 - acc: 0.7181 - val_loss: 1.5340 - val_acc: 0.4195
Epoch 11/50
9/9 [==============================] - 5s 614ms/step - loss: 0.9982 - acc: 0.7617 - val_loss: 1.5188 - val_acc: 0.4273
Epoch 12/50
9/9 [==============================] - 5s 608ms/step - loss: 0.9221 - acc: 0.8066 - val_loss: 1.5070 - val_acc: 0.4301
Epoch 13/50
9/9 [==============================] - 5s 610ms/step - loss: 0.8553 - acc: 0.8398 - val_loss: 1.5003 - val_acc: 0.4320
Epoch 14/50
9/9 [==============================] - 5s 610ms/step - loss: 0.7916 - acc: 0.8659 - val_loss: 1.4927 - val_acc: 0.4351
Epoch 15/50
9/9 [==============================] - 5s 611ms/step - loss: 0.7340 - acc: 0.8958 - val_loss: 1.4820 - val_acc: 0.4386
Epoch 16/50
9/9 [==============================] - 5s 613ms/step - loss: 0.6735 - acc: 0.9167 - val_loss: 1.4827 - val_acc: 0.4396
Epoch 17/50
9/9 [==============================] - 5s 618ms/step - loss: 0.6216 - acc: 0.9290 - val_loss: 1.4768 - val_acc: 0.4415
Epoch 18/50
9/9 [==============================] - 5s 620ms/step - loss: 0.5777 - acc: 0.9455 - val_loss: 1.4771 - val_acc: 0.4472
Epoch 19/50
9/9 [==============================] - 5s 622ms/step - loss: 0.5341 - acc: 0.9546 - val_loss: 1.4750 - val_acc: 0.4437
Epoch 20/50
9/9 [==============================] - 5s 623ms/step - loss: 0.4981 - acc: 0.9627 - val_loss: 1.4683 - val_acc: 0.4551
Epoch 21/50
9/9 [==============================] - 5s 623ms/step - loss: 0.4641 - acc: 0.9701 - val_loss: 1.4693 - val_acc: 0.4502
Epoch 22/50
9/9 [==============================] - 5s 623ms/step - loss: 0.4207 - acc: 0.9785 - val_loss: 1.4701 - val_acc: 0.4495
Epoch 23/50
9/9 [==============================] - 5s 623ms/step - loss: 0.3972 - acc: 0.9787 - val_loss: 1.4679 - val_acc: 0.4490
Epoch 24/50
9/9 [==============================] - 5s 621ms/step - loss: 0.3745 - acc: 0.9837 - val_loss: 1.4680 - val_acc: 0.4500
Epoch 25/50
9/9 [==============================] - 5s 619ms/step - loss: 0.3423 - acc: 0.9898 - val_loss: 1.4648 - val_acc: 0.4533
Epoch 26/50
9/9 [==============================] - 5s 619ms/step - loss: 0.3199 - acc: 0.9926 - val_loss: 1.4677 - val_acc: 0.4485
Epoch 27/50
9/9 [==============================] - 5s 617ms/step - loss: 0.2999 - acc: 0.9918 - val_loss: 1.4705 - val_acc: 0.4544
Epoch 28/50
9/9 [==============================] - 5s 618ms/step - loss: 0.2808 - acc: 0.9941 - val_loss: 1.4687 - val_acc: 0.4540
Epoch 29/50
9/9 [==============================] - 5s 615ms/step - loss: 0.2615 - acc: 0.9954 - val_loss: 1.4723 - val_acc: 0.4540
Epoch 30/50
9/9 [==============================] - 5s 617ms/step - loss: 0.2407 - acc: 0.9972 - val_loss: 1.4720 - val_acc: 0.4564
Epoch 31/50
9/9 [==============================] - 5s 614ms/step - loss: 0.2316 - acc: 0.9970 - val_loss: 1.4712 - val_acc: 0.4583
Epoch 32/50
9/9 [==============================] - 5s 615ms/step - loss: 0.2180 - acc: 0.9983 - val_loss: 1.4735 - val_acc: 0.4549
Epoch 33/50
9/9 [==============================] - 5s 616ms/step - loss: 0.2079 - acc: 0.9985 - val_loss: 1.4733 - val_acc: 0.4541
Epoch 34/50
9/9 [==============================] - 5s 616ms/step - loss: 0.1990 - acc: 0.9978 - val_loss: 1.4636 - val_acc: 0.4568
Epoch 35/50
9/9 [==============================] - 5s 616ms/step - loss: 0.1967 - acc: 0.9989 - val_loss: 1.4755 - val_acc: 0.4557
Epoch 36/50
9/9 [==============================] - 5s 616ms/step - loss: 0.1887 - acc: 0.9996 - val_loss: 1.4836 - val_acc: 0.4547
Epoch 37/50
9/9 [==============================] - 5s 618ms/step - loss: 0.1813 - acc: 0.9998 - val_loss: 1.4767 - val_acc: 0.4567
Epoch 38/50
9/9 [==============================] - 5s 617ms/step - loss: 0.1787 - acc: 0.9993 - val_loss: 1.4720 - val_acc: 0.4570
Epoch 39/50
9/9 [==============================] - 5s 617ms/step - loss: 0.1770 - acc: 0.9996 - val_loss: 1.4793 - val_acc: 0.4568
Epoch 40/50
9/9 [==============================] - 5s 618ms/step - loss: 0.1772 - acc: 1.0000 - val_loss: 1.4763 - val_acc: 0.4563
Epoch 41/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1669 - acc: 0.9996 - val_loss: 1.4780 - val_acc: 0.4563
Epoch 42/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1624 - acc: 1.0000 - val_loss: 1.4818 - val_acc: 0.4552
Epoch 43/50
9/9 [==============================] - 5s 617ms/step - loss: 0.1624 - acc: 0.9993 - val_loss: 1.4796 - val_acc: 0.4571
Epoch 44/50
9/9 [==============================] - 5s 618ms/step - loss: 0.1668 - acc: 1.0000 - val_loss: 1.4828 - val_acc: 0.4605
Epoch 45/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1597 - acc: 0.9998 - val_loss: 1.4771 - val_acc: 0.4578
Epoch 46/50
9/9 [==============================] - 5s 620ms/step - loss: 0.1622 - acc: 1.0000 - val_loss: 1.4761 - val_acc: 0.4579
Epoch 47/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1579 - acc: 1.0000 - val_loss: 1.4817 - val_acc: 0.4549
Epoch 48/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1561 - acc: 1.0000 - val_loss: 1.4790 - val_acc: 0.4594
Epoch 49/50
9/9 [==============================] - 5s 620ms/step - loss: 0.1604 - acc: 1.0000 - val_loss: 1.4779 - val_acc: 0.4584
Epoch 50/50
9/9 [==============================] - 5s 619ms/step - loss: 0.1621 - acc: 0.9993 - val_loss: 1.4748 - val_acc: 0.4593

```
</div>
Pretty bad results!


```python
pt_eval_model = get_eval_model(
    img_size=96,
    backbone=contrastive_model.backbone,
    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
    trainable=False,
    lr=30.0,
)
pt_eval_model.summary()
pt_history = pt_eval_model.fit(
    eval_train_ds,
    batch_size=BATCH_SIZE,
    epochs=TEST_EPOCHS,
    steps_per_epoch=TEST_STEPS_PER_EPOCH,
    validation_data=eval_val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
)
```

<div class="k-default-codeblock">
```
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 eval_input (InputLayer)     [(None, 96, 96, 3)]       0         
                                                                 
 similarity_model (Similarit  (None, 512)              11186112  
 yModel)                                                         
                                                                 
 dense_1 (Dense)             (None, 10)                5130      
                                                                 
=================================================================
Total params: 11,191,242
Trainable params: 5,130
Non-trainable params: 11,186,112
_________________________________________________________________
Epoch 1/50
9/9 [==============================] - 4s 386ms/step - loss: 16.0121 - acc: 0.3008 - val_loss: 16.3010 - val_acc: 0.4517
Epoch 2/50
9/9 [==============================] - 3s 348ms/step - loss: 14.4937 - acc: 0.4477 - val_loss: 10.4881 - val_acc: 0.4498
Epoch 3/50
9/9 [==============================] - 3s 351ms/step - loss: 12.3091 - acc: 0.4848 - val_loss: 8.9369 - val_acc: 0.5239
Epoch 4/50
9/9 [==============================] - 3s 350ms/step - loss: 8.7901 - acc: 0.5302 - val_loss: 7.8179 - val_acc: 0.5310
Epoch 5/50
9/9 [==============================] - 3s 350ms/step - loss: 6.8679 - acc: 0.5588 - val_loss: 7.6582 - val_acc: 0.5405
Epoch 6/50
9/9 [==============================] - 3s 349ms/step - loss: 6.4141 - acc: 0.5269 - val_loss: 5.9548 - val_acc: 0.5445
Epoch 7/50
9/9 [==============================] - 3s 350ms/step - loss: 6.0155 - acc: 0.5315 - val_loss: 7.4021 - val_acc: 0.4644
Epoch 8/50
9/9 [==============================] - 3s 351ms/step - loss: 6.9686 - acc: 0.5113 - val_loss: 7.0094 - val_acc: 0.4923
Epoch 9/50
9/9 [==============================] - 3s 351ms/step - loss: 7.2035 - acc: 0.5202 - val_loss: 5.9155 - val_acc: 0.5397
Epoch 10/50
9/9 [==============================] - 3s 351ms/step - loss: 5.7862 - acc: 0.5417 - val_loss: 6.5536 - val_acc: 0.5142
Epoch 11/50
9/9 [==============================] - 3s 351ms/step - loss: 6.3274 - acc: 0.5349 - val_loss: 4.5757 - val_acc: 0.5767
Epoch 12/50
9/9 [==============================] - 3s 351ms/step - loss: 3.7128 - acc: 0.5770 - val_loss: 3.8968 - val_acc: 0.5639
Epoch 13/50
9/9 [==============================] - 3s 351ms/step - loss: 5.0269 - acc: 0.5456 - val_loss: 6.0357 - val_acc: 0.5333
Epoch 14/50
9/9 [==============================] - 3s 351ms/step - loss: 6.2850 - acc: 0.5193 - val_loss: 7.4535 - val_acc: 0.4865
Epoch 15/50
9/9 [==============================] - 3s 351ms/step - loss: 5.9532 - acc: 0.5514 - val_loss: 3.7321 - val_acc: 0.5846
Epoch 16/50
9/9 [==============================] - 3s 351ms/step - loss: 4.0213 - acc: 0.5720 - val_loss: 4.9974 - val_acc: 0.5618
Epoch 17/50
9/9 [==============================] - 3s 351ms/step - loss: 5.1417 - acc: 0.5395 - val_loss: 4.4524 - val_acc: 0.5168
Epoch 18/50
9/9 [==============================] - 3s 351ms/step - loss: 5.1963 - acc: 0.5330 - val_loss: 3.0755 - val_acc: 0.6182
Epoch 19/50
9/9 [==============================] - 3s 351ms/step - loss: 3.9321 - acc: 0.5634 - val_loss: 4.3298 - val_acc: 0.5758
Epoch 20/50
9/9 [==============================] - 3s 351ms/step - loss: 3.4384 - acc: 0.5870 - val_loss: 2.9364 - val_acc: 0.5912
Epoch 21/50
9/9 [==============================] - 3s 350ms/step - loss: 2.7146 - acc: 0.5968 - val_loss: 2.6873 - val_acc: 0.6196
Epoch 22/50
9/9 [==============================] - 3s 351ms/step - loss: 2.4992 - acc: 0.6159 - val_loss: 2.2900 - val_acc: 0.6157
Epoch 23/50
9/9 [==============================] - 3s 351ms/step - loss: 2.2021 - acc: 0.6170 - val_loss: 2.1647 - val_acc: 0.6010
Epoch 24/50
9/9 [==============================] - 3s 350ms/step - loss: 2.2598 - acc: 0.6126 - val_loss: 2.4756 - val_acc: 0.6229
Epoch 25/50
9/9 [==============================] - 3s 351ms/step - loss: 2.2574 - acc: 0.6096 - val_loss: 2.9688 - val_acc: 0.5829
Epoch 26/50
9/9 [==============================] - 3s 350ms/step - loss: 2.4860 - acc: 0.5940 - val_loss: 2.3797 - val_acc: 0.6030
Epoch 27/50
9/9 [==============================] - 3s 351ms/step - loss: 2.1329 - acc: 0.6202 - val_loss: 1.9786 - val_acc: 0.6289
Epoch 28/50
9/9 [==============================] - 3s 351ms/step - loss: 1.7937 - acc: 0.6547 - val_loss: 2.2189 - val_acc: 0.6185
Epoch 29/50
9/9 [==============================] - 3s 351ms/step - loss: 1.7103 - acc: 0.6480 - val_loss: 1.6566 - val_acc: 0.6339
Epoch 30/50
9/9 [==============================] - 3s 350ms/step - loss: 1.4796 - acc: 0.6634 - val_loss: 1.6646 - val_acc: 0.6450
Epoch 31/50
9/9 [==============================] - 3s 350ms/step - loss: 1.3734 - acc: 0.6721 - val_loss: 1.5819 - val_acc: 0.6336
Epoch 32/50
9/9 [==============================] - 3s 350ms/step - loss: 1.3773 - acc: 0.6799 - val_loss: 1.5930 - val_acc: 0.6382
Epoch 33/50
9/9 [==============================] - 3s 350ms/step - loss: 1.3305 - acc: 0.6660 - val_loss: 1.6665 - val_acc: 0.6299
Epoch 34/50
9/9 [==============================] - 3s 351ms/step - loss: 1.2551 - acc: 0.6901 - val_loss: 1.5166 - val_acc: 0.6563
Epoch 35/50
9/9 [==============================] - 3s 351ms/step - loss: 1.2091 - acc: 0.6923 - val_loss: 1.5088 - val_acc: 0.6588
Epoch 36/50
9/9 [==============================] - 3s 351ms/step - loss: 1.1862 - acc: 0.7051 - val_loss: 1.4047 - val_acc: 0.6600
Epoch 37/50
9/9 [==============================] - 3s 351ms/step - loss: 1.1528 - acc: 0.7018 - val_loss: 1.4526 - val_acc: 0.6523
Epoch 38/50
9/9 [==============================] - 3s 350ms/step - loss: 1.1019 - acc: 0.7023 - val_loss: 1.3786 - val_acc: 0.6595
Epoch 39/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0913 - acc: 0.7029 - val_loss: 1.3847 - val_acc: 0.6670
Epoch 40/50
9/9 [==============================] - 3s 350ms/step - loss: 1.1141 - acc: 0.7072 - val_loss: 1.3093 - val_acc: 0.6659
Epoch 41/50
9/9 [==============================] - 3s 350ms/step - loss: 1.1157 - acc: 0.7031 - val_loss: 1.3882 - val_acc: 0.6574
Epoch 42/50
9/9 [==============================] - 3s 351ms/step - loss: 1.1021 - acc: 0.7036 - val_loss: 1.3534 - val_acc: 0.6545
Epoch 43/50
9/9 [==============================] - 3s 351ms/step - loss: 1.0547 - acc: 0.6986 - val_loss: 1.3103 - val_acc: 0.6655
Epoch 44/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0417 - acc: 0.7083 - val_loss: 1.2934 - val_acc: 0.6636
Epoch 45/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0390 - acc: 0.7109 - val_loss: 1.2921 - val_acc: 0.6684
Epoch 46/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0218 - acc: 0.7153 - val_loss: 1.2974 - val_acc: 0.6658
Epoch 47/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0281 - acc: 0.7155 - val_loss: 1.2944 - val_acc: 0.6690
Epoch 48/50
9/9 [==============================] - 3s 351ms/step - loss: 1.0101 - acc: 0.7179 - val_loss: 1.2812 - val_acc: 0.6698
Epoch 49/50
9/9 [==============================] - 3s 351ms/step - loss: 1.0269 - acc: 0.7099 - val_loss: 1.3000 - val_acc: 0.6673
Epoch 50/50
9/9 [==============================] - 3s 350ms/step - loss: 1.0139 - acc: 0.7151 - val_loss: 1.2850 - val_acc: 0.6685

```
</div>
Far superior results!
