import os
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import cv2
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

import keras2caffe

#TensorFlow backend uses all GPU memory by default, so we need limit
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

#converting
keras_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
keras_model.summary()

img_keras = image.load_img('bear.jpg', target_size=(224, 224))
data_keras = image.img_to_array(img_keras)
data_keras = np.expand_dims(data_keras, axis=0)
data_keras -= 128

keras2caffe.convert(keras_model, 'VGG16.prototxt', 'VGG16.caffemodel')

#testing the model

#caffe.set_mode_gpu()
net_ref  = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
net  = caffe.Net('VGG16.prototxt', 'VGG16.caffemodel', caffe.TEST)
#for layer_name, param in net.params.iteritems():
#    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

img = cv2.imread('bear.jpg')
img = cv2.resize(img, (224, 224))
img = img[...,::-1]  #RGB 2 BGR

data = np.array(img, dtype=np.float32)
data = data.transpose((2, 0, 1))
data.shape = (1,) + data.shape

data -= 128

#verify
pred = keras_model.predict(data_keras)[0]
prob = np.max(pred)
cls = pred.argmax()
lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]

out = net.forward_all(data = data)
pred = out['predictions']
prob = np.max(pred)
cls = pred.argmax()
lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]

out = net_ref.forward_all(data = data)
pred = out['prob']
prob = np.max(pred)
cls = pred.argmax()
lines=open('synset_words.txt').readlines()
print prob, cls, lines[cls]
