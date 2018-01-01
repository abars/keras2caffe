from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model

import matplotlib.pyplot as plt

import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import cv2
import numpy as np

import keras2caffe

def vis_square(data):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))
               + ((0, 0),) * (data.ndim - 3))
    data = np.pad(data, padding, mode='constant', constant_values=1)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data);
    plt.axis('off')

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
    raise Exception('keras2caffe shoud be channel_last')
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#caffe.set_mode_gpu()
net  = caffe.Net('mnist.prototxt', 'mnist.caffemodel', caffe.TEST)
#for layer_name, blob in net.blobs.iteritems():
#    print (layer_name + '\t' + str(blob.data.shape))
#filters = net.blobs['conv2d_1'].data[0, :36]
#vis_square(filters.transpose(0, 2, 3, 1))

#verify
for i in range(1):
	data = x_train[i]
	data.shape = (1,) + data.shape
	pred = keras_model.predict(data)

	#get_1st_layer_output = K.function([keras_model.layers[0].input],
    #                              [keras_model.layers[0].output])
	#layer_output = get_1st_layer_output([data,])
	#print(keras_model.layers[0].name)
	#print(layer_output[0])

	#get_1st_layer_output = K.function([keras_model.layers[1].input],
    #                              [keras_model.layers[1].output])
	#layer_output = get_1st_layer_output([data,])
	#print(keras_model.layers[1].name)

	#print(pred)

	print("keras Class is: " + str(np.argmax(pred)))
	print("Certainty is: " + str(pred[0][np.argmax(pred)]))
	print("Refernce is: "+str(y_train[i]))

print(" ")

for i in range(1):
	data = x_train[i]
	data = data.transpose(2,0,1)
	data.shape = (1,) + data.shape
	
	out = net.forward_all(data=data)

	#print(net.blobs['data'].data[0])
	#print(net.blobs['conv2d_1'].data[0])

	pred = out['dense_2']
	#print(pred)

	print("Caffe Class is: " + str(np.argmax(pred)))
	print("Certainty is: " + str(pred[0][np.argmax(pred)]))
	print("Refernce is: "+str(y_train[i]))
