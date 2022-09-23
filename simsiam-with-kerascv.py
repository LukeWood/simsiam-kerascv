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
PRE_TRAIN_EPOCHS = 800
VAL_STEPS_PER_EPOCH = 20
WEIGHT_DECAY = 5e-4
INIT_LR = 3e-2 * int(BATCH_SIZE / 256)
WARMUP_LR = 0.0
WARMUP_STEPS = 0
DIM = 2048

"""
## Data Loading

Next, we will load the STL-10 dataset.  STL-10 is a dataset consisting of 100k unlabelled
images, 5k labelled training images, and 10k labelled test images.  Due to this distribution,
STL-10 is commonly used as a benchmark for contrastive learning models.

First lets load our unlabelled data
"""
train_ds = tfds.load("stl10", split="unlabelled")
train_ds = train_ds.map(
    lambda entry: entry["image"], num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.shuffle(buffer_size=8 * BATCH_SIZE, reshuffle_each_iteration=True)

"""
Next, we need to prepare some labelled samples.
This is done so that TensorFlow similarity can probe the learned embedding to ensure
that the model is learning appropriately.
"""

(x_raw_train, y_raw_train), ds_info = tfds.load(
    "stl10", split="train", as_supervised=True, batch_size=-1, with_info=True
)
x_test, y_test = tfds.load(
    "stl10",
    split="test",
    as_supervised=True,
    batch_size=-1,
)

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
    return tf.convert_to_tensor(np.array(x)), tf.convert_to_tensor(np.array(y))


x_query, y_query = create_split(query_idxs)
x_index, y_index = create_split(index_idxs)
x_val, y_val = create_split(val_idxs)
x_train, y_train = create_split(train_idxs)

PRE_TRAIN_STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

print(
    tabulate(
        [
            ["train", x_train.shape, y_train.shape],
            ["val", x_val.shape, y_val.shape],
            ["query", x_query.shape, y_query.shape],
            ["index", x_index.shape, y_index.shape],
            ["test", x_test.shape, y_test.shape],
        ],
        headers=["Examples", "Labels"],
    )
)

"""
## Augmentations
Self-supervised networks require at least two augmented "views" of each example. This can be created using a DataSet and an augmentation function. The DataSet treats each example in the batch as its own class and then the augment function produces two separate views for each example.

This means the resulting batch will yield tuples containing the two views, i.e.,
Tuple[(BATCH_SIZE, 32, 32, 3), (BATCH_SIZE, 32, 32, 3)].

TensorFlow Similarity provides several random augmentation functions, and here we combine augmenters from the simCLR module to replicate the augmentations used in simsiam.
"""

"""
## Augmentation

Now that we have all of our datasets produced, we can move on to dataset augmentation.
Using KerasCV, it is trivial to construct an augmenter that performs as the one
described in the original SimSiam paper.  Lets do that below.
"""

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

"""
Next, lets pass our images through this pipeline.
Note that KerasCV supports batched augmentation, so batching before
augmentation dramatically improves performance

"""


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

"""
Lets visualize our pairs using the `tfsim.visualization` utility package.
"""
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

"""
## Model Creation

Now that our data and augmentation pipeline is setup, we can move on to
constructing the contrastive learning pipeline.  First, lets produce a backbone.
For this task, we will use a KerasCV ResNet18 model as the backbone.
"""


def get_backbone(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = keras_cv.models.ResNet18(
        input_shape=input_shape,
        include_rescaling=True,
        include_top=False,
    )(x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    return tfsim.models.SimilarityModel(inputs, x, name="resnet18sim")


backbone = get_backbone((96, 96, 3))
backbone.summary()

"""
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
"""


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


"""
Finally, we must construct the predictor.  The predictor is used in SimSiam, and is a
simple stack of two MLP layers, containing a bottleneck in the hidden layer.
"""


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


"""
## Training

First, we need to initialize our training model, loss, and optimizer.
"""
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

"""
Next we can compile the model the same way you compile any other Keras model.
"""

contrastive_model.compile(
    optimizer=optimizer,
    loss=loss,
)

"""
We track the training using several callbacks.

* **EvalCallback** creates an index at the end of each epoch and provides a proxy for the nearest neighbor matching classification using `binary_accuracy`.
* **TensordBoard** and **ModelCheckpoint** are provided for tracking the training progress.
"""

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

"""
All that is left to do is run fit()!
"""

history = contrastive_model.fit(
    train_ds,
    epochs=PRE_TRAIN_EPOCHS,
    steps_per_epoch=PRE_TRAIN_STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
    callbacks=callbacks,
)


"""
## Plotting and Evaluation
"""

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


"""
## Fine Tuning on the Labelled Data
TODO
"""

eval_augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomCropAndResize((96, 96), crop_area_factor=(0.2, 1.0)),
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

"""
## Benchmark Against a Naive Model
"""


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
    img_size=CIFAR_IMG_SIZE,
    backbone=get_backbone(CIFAR_IMG_SIZE, DIM),
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

pt_history = pt_eval_model.fit(
    eval_train_ds,
    batch_size=BATCH_SIZE,
    epochs=TEST_EPOCHS,
    steps_per_epoch=TEST_STEPS_PER_EPOCH,
    validation_data=eval_val_ds,
    validation_steps=VAL_STEPS_PER_EPOCH,
)
pt_eval_model = get_eval_model(
    img_size=CIFAR_IMG_SIZE,
    backbone=contrastive_model.backbone,
    total_steps=TEST_EPOCHS * TEST_STEPS_PER_EPOCH,
    trainable=False,
    lr=30.0,
)
pt_eval_model.summary()
