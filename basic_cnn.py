# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:10:49 2021

@author: Manasi
"""
## Digit Recognizer with tensorflow
## import required packages
import tensorflow
tensorflow.__version__

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
%matplotlib inline

#------------------------------------------------------------------------------
#                       Load Data                                              
#--------------------------------------------------------------------------
## Load data
train=pd.read_csv('C:\\Users\\rajhi\\Manasi\\Kaggle\\MNIST\\data\\train\\train.csv')
test=pd.read_csv('C:\\Users\\rajhi\\Manasi\\Kaggle\\MNIST\\data\\test\\test.csv')

## check train data
train.info()
train.head()

## Check train data
test.info()

'''
train Data has 42000 image details.and test data has 28000 images
Each row represents one label/digit.
image/pixel information is avaiable in columns.
each pixel value is in integer from 0-255
'''
### check unique labels
####### train #############
X_train=train.drop('label', axis=1)
y_train=train[['label']]

y_train.value_counts(normalize=True)
y_train.value_counts().plot(kind='bar')

'''
10 unique values almost equal number of images.
'''
#------------------------------------------------------------------------------
#                       Preprocess Data                              
#--------------------------------------------------------------------------
## 2.1 :  Reshape
'''
Sequential model in tensorflow.keras expects data to be in the format (n_e, n_h, n_w, n_c)
n_e= number of examples, n_h = height, n_w = width, n_c = number of channels
do not reshape labels
'''
X_train_reshape= X_train.values.reshape(-1, 28, 28, 1)
X_test_reshape= test.values.reshape(-1, 28, 28, 1)

## 2.2 Normalize data images
'''
1. Must Normalize images for neural network
2.we can achieve this by dividing the RGB codes with 255 (which is the maximum RGB code minus the minimum RGB code)
3. normalize X_train and X_test
4. make sure that the values are float so that we can get decimal points after division
'''
X_train_norm = X_train_reshape.astype('float32')
X_test_norm = X_test_reshape.astype('float32')
X_train_norm /= 255
X_test_norm /= 255

## Checking shape of images
print("X_train shape:", X_train.shape)
print("Images in X_train:", X_train_norm.shape)
print("Images in X_test:", X_test_norm.shape[0])
print("Max value in X_train:", X_train_norm.max())
print("Min value in X_train:", X_train_norm.min())


### check images
plt.figure(5, figsize=(10,10))
for i in range(0,5):
    plt.subplot(1,5,i+1)
    print("Label: {}".format(y_train['label'][i]))
    plt.imshow(X_train_reshape[i], cmap='gray')
    plt.xlabel(y_train['label'][i])
    
## 2.3 One hot encoding for class vector
'''
convert class vectors (integers) to binary class matrix
convert y_train 
number of classes: length of unique values of class vector
due to multiclass classification we will be using categorical_crossentropy as loss
'''
classes= y_train['label'].unique()

y_train_cat = to_categorical(y_train['label'], num_classes=len(classes))

print("Shape of y_train:", y_train_cat.shape)
print("original value of y_train:", y_train['label'][0])
print("One value of y_train:", y_train_cat[0])
#------------------------------------------------------------------------------
#                       Build Model                             
#--------------------------------------------------------------------------
x_train, x_test, y_train, y_test= train_test_split(X_train_norm,y_train_cat, test_size=0.2,stratify=y_train_cat)

## 3.1 Define model : Architecture of the model
from keras import models, layers,optimizers
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
## output shape : 28-3+1 = 26, 26 x26 images size with 32 channels/filters
## Learnable Parameters =((width of Filter X height of Filter X depth of Filter + 1) X Number of Filters) 
## calculate learnable parameters : (3*3*1+1)*32= 320 # note : here depth is 1 as image is grayscale and +1 is for bais parameter

model.add(layers.Flatten())
model.add(layers.Dense(10, activation="softmax"))
model.summary()

## 3.2 Compile model
from keras import optimizers

model.compile(loss ="categorical_crossentropy", optimizer = "adam",
metrics =['accuracy'])
  
model.fit(x=x_train,y=y_train,batch_size=32, epochs = 10,validation_data = (x_test,y_test))

## 3.2 Checking outout of each layer
from keras import models
layers_output=[]
layers_output=[layer.output for layer in model.layers[:3]]
activation_model= models.Model(inputs=model.input, outputs=layers_output)
img_tensor = np.expand_dims(x_train[0], axis = 0)
activations = activation_model.predict(img_tensor)

# Getting Activations of first layer
first_layer_activation = activations[0]
# shape of first layer activation
print(first_layer_activation.shape)

# 6th channel of the image after first layer of convolution is applied
plt.figure(36, figsize=(7,7))
for i in range(0,32):
    plt.subplot(6,6,i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap ='gray')


model.evaluate(x_test, y_test)

# predict results
results = model.predict(X_test_norm)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_cnn.csv",index=False)

#### validation

plt.figure(5, figsize=(10,10))
for i in range(15,19):
    plt.subplot(15,5,i+1)
    print("Label: {}".format(results[i]))
    plt.imshow(X_test_norm[i], cmap='gray')
    plt.xlabel(results[i])
    
os.getcwd()