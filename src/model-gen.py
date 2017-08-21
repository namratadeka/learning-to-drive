from __future__ import division
import os
import csv
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## Reading csv file.
samples = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)


## Reading the images from multiple cameras.
def read_data(samples):
	images = []
	measurements = []
	correction = [0, 0.5, -0.5]		# Correction biases for center, left and right images respectively.
	for line in samples:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = './IMG/'+filename
			image = cv2.imread(current_path)
			images.append(image)
			measurement = float(line[3])+correction[i]
			measurements.append(measurement)
	return images,measurements

## Augmenting available training data to reduce bias of left turns by flipping them.
def augment(images, measurements):
	augmented_images, augmented_measurements = [], []
	for image, measurement in zip(images, measurements):
		augmented_images.append(image)
		augmented_measurements.append(measurement)
		augmented_images.append(cv2.flip(image, 1))
		augmented_measurements.append(-1.0 * measurement)
	return augmented_images,augmented_measurements

train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)

def generator(samples, batch_size=32):
	n = len(samples)
	while True:
		for offset in range(0,n,batch_size):
			images,measurements = read_data(samples[offset:offset+batch_size])
			images,measurements = augment(images,measurements)

			X_train = np.array(images)
			y_train = np.array(measurements)

			yield [X_train,y_train]

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

model = Sequential()
## NVIDIA model
model.add(Lambda(lambda x: x / 255.0  - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(100))
model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
##
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
filepath = "./tmp/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=1, monitor='val_acc', save_best_only=False)
model.fit_generator(train_generator, 
					steps_per_epoch=len(train_samples),
					validation_data=validation_generator,
					validation_steps=len(validation_samples),
					epochs=20,
					callbacks=[checkpointer])
model.save('model-gen.h5')
