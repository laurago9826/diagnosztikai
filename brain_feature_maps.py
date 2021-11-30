from keras.models import Sequential
import tensorflow as tf

#import tensorflow_datasets as tfds

#tf.enable_eager_execution()
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt

path_root = "D:/temp/kozlobeadando"
path_test = path_root + "/data/Training"

model = tf.keras.models.load_model(path_root + '/model/brain_cancer_model')

CATEGORIES = ["pituitary_tumor", "no_tumor", "meningioma_tumor", "glioma_tumor"]
IMG_SIZE = 200
class_idx = 3

layer_outputs = [layer.output for layer in model.layers]
feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

def create_image_from_array(image):
  max1 = np.amax(image)
  min1 = np.amin(image)
  max2 = 255
  min2 = 0
  new_val_mul = (max2 - min2) / (max1 - min1)
  new_array = np.uint8(image * new_val_mul)
  img = Image.fromarray(new_array, 'L')
  return img

def createFeatureMapData():
  feature_map_images = []
  for category in CATEGORIES:
    path = os.path.join(path_test, category)
    file = os.listdir(path)[1]
    img_array = cv2.imread(os.path.join(path,file))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    feature_map_images.append(new_array)
  return feature_map_images

feature_map_images = [createFeatureMapData()[class_idx]]

X = np.array(feature_map_images).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X.astype('float32')
X /= 255

feature_maps = [fmap for fmap in feature_map_model.predict(X) if (len(fmap.shape) == 4)]
layer_names = [layer.name for layer in model.layers]
num_images = 0



for feature_map in feature_maps:
  if len(feature_map.shape) == 4:
    num_images += feature_map.shape[3]

for i, item in enumerate(zip(layer_names, feature_maps)):
  layer_name = item[0]
  feature_map = item[1]
  fmap_size = int(feature_map.shape[3])
  for j in range(fmap_size):
    fmap = feature_map[0,:,:,j]
    img = create_image_from_array(fmap)
    img.save('feature_maps/'+ CATEGORIES[class_idx] + "/" + str(i * fmap_size + j + 1)+'.png')
    
