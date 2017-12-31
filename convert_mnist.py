from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model

import caffe
import cv2
import numpy as np

import keras2caffe

INLINE_ACTIVATION=False
if(INLINE_ACTIVATION==False):
	keras_model = load_model("mnist.hdf5")
else:
	keras_model = load_model("mnist_inline.hdf5")
keras_model.summary()

keras2caffe.convert(keras_model, 'mnist.prototxt', 'mnist.caffemodel')

# input image dimensions
img_rows, img_cols = 28, 28

#load test data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

for i in range(5):
	data = x_train[i]
	data.shape = (1,) + data.shape
	pred = keras_model.predict(data)
	print("Class is: " + str(np.argmax(pred)))
	print("Certainty is: " + str(pred[0][np.argmax(pred)])+" vs "+str(y_train[i]))

#caffe.set_mode_gpu()
net  = caffe.Net('mnist.prototxt', 'mnist.caffemodel', caffe.TEST)

for i in range(5):
	data = x_train[i]
	data = data.transpose(2,0,1)
	net.blobs['data'].data[...] = data
	out = net.forward()
	preds = out['dense_2']
	print("Class is: " + str(np.argmax(pred)))
	print("Certainty is: " + str(pred[0][np.argmax(pred)])+" vs "+str(y_train[i]))



