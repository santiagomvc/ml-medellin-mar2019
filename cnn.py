#%%
import scipy
from PIL import Image
from scipy import ndimage
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd
import tensorflow as tf


path = 'C:\\Users\\santi\\Google Drive\\ml-medellin-mar2019'
# defining folders path for training
train = path + '\\train'

#read info from training
X = []
Y = []
for i in os.listdir(train):
    label = i
    for j in  os.listdir(train + '\\' + i):
        fname = train + '\\' + i + '\\' + j
        image = np.array(ndimage.imread(fname, flatten=False))
        reshaped_image = scipy.misc.imresize(image, size=(64,64))
        X.append(reshaped_image)
        Y.append(int(label))


X = np.array(X)
Y = np.array(Y)

X = X/255

# separando dev and test
dev_size = math.ceil(0.15 * X.shape[0])
dev_index = np.random.randint(0, X.shape[0], dev_size)
X_test = X[dev_index, :, :, :]
Y_test = Y[dev_index] 
X_train = np.delete(X, dev_index, axis = 0)
Y_train = np.delete(Y, dev_index, axis = 0)

#%%
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  #print(pool2.shape)
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=43)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions['classes'])

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#%%
# Create the Estimator
signal_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=path + '\\signal_convnet_model')


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=Y_train,
    batch_size=64,
    num_epochs=None,
    shuffle=False)

# # train one step and display the probabilties
# signal_classifier.train(
#     input_fn=train_input_fn,
#     steps=1,
#     hooks=[logging_hook])

signal_classifier.train(input_fn=train_input_fn, steps=5)


#%%
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=Y_test,
    num_epochs=1,
    shuffle=False)

eval_results = signal_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

#%%
#read info from training
test = path + '\\test_files'
X_prove = []
X_prove_name = []
for i in os.listdir(test):
    fname = test + '\\' + i
    image = np.array(ndimage.imread(fname, flatten=False))
    reshaped_image = scipy.misc.imresize(image, size=(64,64))
    X_prove.append(reshaped_image)
    X_prove_name.append(i)
 #   Y_train.append(float(label))

X_prove = np.array(X_prove)
X_prove = X_prove/255.

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_prove},
    num_epochs=1,
    shuffle=False)

test_results = np.array(list(signal_classifier.predict(input_fn=test_input_fn)))
print(test_results)

#%%
df_results = pd.DataFrame({'file_id': X_prove_name, 'label': test_results})
test_csv = pd.read_csv(path + '\\test.csv')
test_csv = test_csv.drop('label', axis = 1)
definitive_results = pd.merge(test_csv, df_results, on = 'file_id', how = 'outer')
definitive_results.to_csv(path + '\\test_results.csv', index=False)
#%%