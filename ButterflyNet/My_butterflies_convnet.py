import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import random

#allow GPU memory growth because TF 2.1 and CUDNN aren't getting along right now
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

with open('../../KagglesData/Butterflies/fgvc_fg_training.json','r') as anno_train:
    train = json.load(anno_train)
with open('../../KagglesData/Butterflies/fgvc_fg_testing.json','r') as anno_test:
    test = json.load(anno_test)

train.keys()
test.keys()
test_df = pd.DataFrame()
test_df = test_df.append(test['images'], ignore_index=True)
train_df = pd.DataFrame()
train_df = train_df.append(train['images'], ignore_index=True)
train_df_anno = pd.DataFrame()
train_df_anno = train_df_anno.append(train['annotations'], ignore_index=True)
train_df['category_id'] = train_df_anno['category_id']
test_df.head()
train_df.head()
foldertrain = '../../KagglesData/Butterflies/data-images/training/images/'
foldertest = '../../KagglesData/Butterflies/data-images/testing/images/'

#the images are all different sizes, so we will need to pad them with zeroes around the border to create a CNN. Pad to 600x600 to preserve orientation of features.
#takes dataframe and image directory, returns 600x600x3x(SIZE) ndarray of images. This is very slow, so to avoid looping over the dataframe more times than necessary we combine the function
# to get labels and the function to creat padded images into one.

def getLabelsAndPaddedImages(df, folder, isTrain):
	#slicesize = len(df.index)
	slicesize = 1000
	cat_labels = []
	images = []
	for i in range(slicesize):
		# define subplot
		# define filename
		filename = folder + df.at[i, 'file_name']
		# load image pixels. Images are rgb
		image = imread(filename)
		pad_left = math.ceil((600-df.at[i, 'width'])/2)
		pad_right = math.floor((600-df.at[i, 'width'])/2)
		pad_up = math.ceil((600-df.at[i, 'height'])/2)
		pad_down = math.floor((600-df.at[i, 'height'])/2)
		impad = np.pad(image,((pad_up, pad_down),(pad_left,pad_right),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0)))
		# plot raw rgb pixel data
		#if i < 9:
			#pyplot.subplot(330 + 1 + i)
			#pyplot.imshow(image)
		# show the figure
		#pyplot.show()
		images.append(impad)
		if(i%100 == 0):
			print("got image "),
			print(i)
		if isTrain == 1:
			labels = np.zeros(5419)
			cat_label = df.at[i,'category_id']
			labels[cat_label] = 1
			cat_labels.append(labels)
	if (isTrain == 1):
		return (cat_labels, images)
	else: return images

train_labels_tot,train_img_tot = getLabelsAndPaddedImages(train_df, foldertrain, 1)
test_img = getLabelsAndPaddedImages(test_df, foldertest,0)

c = list(zip(train_labels_tot, train_img_tot))
random.shuffle(c)
train_labels_tot_shuf, train_img_tot_shuf = zip(*c)

train_labels = np.array(train_labels_tot_shuf[:int(len(train_labels_tot_shuf)/2)])
validation_labels = np.array(train_labels_tot_shuf[int(len(train_labels_tot_shuf)/2):])
train_img = np.array(train_img_tot_shuf[:int(len(train_img_tot_shuf)/2)])
validation_img = np.array(train_img_tot_shuf[int(len(train_img_tot_shuf)/2):])

print(len(train_img))

#Now we build the basic convnet. Use Tensorflow for this on an RTX gpu.

# Define a sequential model
input_shape = (600,600,3)
num_classes= 5419
model = Sequential()
# add first convolutional layer
model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
# add second convolutional layer
model.add(Conv2D(32, (5, 5), activation='relu'))
# add one max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# add one dropout layer
model.add(Dropout(0.25))
# add flatten layer
model.add(Flatten())
# add dense layer
model.add(Dense(16, activation='relu'))
# add another dropout layer
model.add(Dropout(0.5))
# add dense layer
model.add(Dense(num_classes, activation='softmax'))
# complile the model and view its architecur
model.compile(loss=keras.losses.categorical_crossentropy,  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print(model.summary())

epochs = 1
batch_size = 1
history = model.fit(train_img,train_labels, batch_size=batch_size,epochs = epochs, validation_data = (validation_img,validation_labels))

print('\n# Generate predictions for test set')
predictions = model.predict(np.array(test_img))
print('predictions shape:', predictions.shape)
predvals = tf.math.argmax(predictions, axis = 1)
print(predvals)
