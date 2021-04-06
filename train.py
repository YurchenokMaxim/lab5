"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""

author = 'Alexander Soroka, soroka.a.m@gmail.com'
copyright = """Copyright 2020 Alexander Soroka"""

import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from math import exp

# Avoid greedy memory allocation to allow shared GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOG_DIR = 'logs'
BATCH_SIZE = 16
NUM_CLASSES = 20
RESIZE_TO = 224
TRAIN_SIZE = 12786

img_rotate =keras.Sequential(
   [
    tf.keras.layers.experimental.preprocessing.RandomRotation(
    factor= 0.5, fill_mode='reflect', interpolation='bilinear',
    seed=None, name=None, fill_value=0.0
)
   ]
)


img_contrast =keras.Sequential(
   [
   tf.keras.layers.experimental.preprocessing.RandomContrast(
   factor=0.7 , seed=None, name=None
)
   ]
)

img_gauss =keras.Sequential(
   [
   tf.keras.layers.GaussianNoise(
    stddev=0.5)
   ]
)
def normalize(image, label):
  return tf.image.per_image_standardization(image), label

def random_crop(image,label):
      return tf.image.random_crop(image,[RESIZE_TO, RESIZE_TO, 3]),label

def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([270, 270]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)


def create_dataset(filenames, batch_size):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .map(random_crop)\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)


def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  model = img_rotate(inputs)
  model = img_contrast(model)
  model = img_gauss(model)
  model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
  model.trainable=False
  model = tf.keras.layers.GlobalAveragePooling2D()(model.output)
  model=tf.image.adjust_brightness(model, 0.3)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(model)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def exp_decay(epoch):
  initial_lrate =0.1
  k =0.4
  lrate = initial_lrate * exp(-k*epoch)
  return lrate

Lrate=LearningRateScheduler(exp_decay)

def unfreeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            print('unfreezed')


def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE)
  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)

  model = build_model()
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )

  log_dir='{}/owl-{}'.format(LOG_DIR, time.time())
  lrate = LearningRateScheduler(exp_decay)
  model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir),lrate

    ]
  )

  unfreeze_model(model)
  model.compile(
    optimizer=tf.optimizers.Adam(1e-7),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )
  model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir)
    ]
  )


if __name__ == '__main__':
    main()
