import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'      #Delete Warnings
import tensorflow as tf
import cv2
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, MaxPooling2D, Convolution2D, Cropping2D, Activation, Conv2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import helper

dir = os.path.dirname(__file__)
images = []
angles = []

for i in range(3):
    datafile = "data_"+ str(i)
    image_path = os.path.join(dir, 'data',datafile,'driving_log.csv')

    angle_adjustment = 0.1
    total_left_angles = 0
    total_right_angles = 0
    total_straight_angles = 0


    with open(image_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_image = cv2.imread(line[0].split('/')[-1])
            center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            images.append(center_image_rgb)
            angles.append(float(line[3]))
            #flipped
            images.append(cv2.flip(center_image_rgb, 1))
            angles.append(-float(line[3]))

            left_image = cv2.imread(line[1].split('/')[-1])
            left_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            images.append(left_image_rgb)
            angles.append(float(line[3])+angle_adjustment)
            #flipped
            images.append(cv2.flip(left_image_rgb, 1))
            angles.append(-(float(line[3])+angle_adjustment))

            right_image = cv2.imread(line[2].split('/')[-1])
            right_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            images.append(right_image_rgb)
            angles.append(float(line[3])-angle_adjustment)
            #flipped
            images.append(cv2.flip(right_image_rgb, 1))
            angles.append(-(float(line[3])-angle_adjustment))

            if(float(line[3]) < -0.15):
                total_left_angles += 1
            elif(float(line[3]) > 0.15):
                total_right_angles += 1
            else:
                total_straight_angles += 1

left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

plt.hist(angles)
plt.show()
print('Total Samples : ', len(images))
# print('Total Images : ', len(lines)*3)
# print('Data Shape : ', data_samples.shape)


# print()
#was 'center','left','right','angle','power','brake'
#now image array, angle array
# print('Preview : ', data_samples[0])

X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

train_samples_size = len(X_train)
validation_samples_size = len(X_val)

total_left_angles = 0
total_right_angles = 0
total_straight_angles = 0

for train_sample in y_train:
    if(float(train_sample) < -0.15):
        total_left_angles += 1
    elif(float(train_sample) > 0.15):
        total_right_angles += 1
    else:
        total_straight_angles += 1

left_to_straight_ratio = 0
right_to_straight_ratio = 0

left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

print()
print('Train Sample Size : ', train_samples_size)
print('Validation Sample Size : ', validation_samples_size)


channels, height, width = 3, 160, 320  # image format
model = Sequential()
model.add(Lambda(lambda x: x / 255. - 0.5,
                 input_shape=(height, width, channels),
                 output_shape=(height, width, channels)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=12, shuffle=True) #, shuffle=True, validation_split=0.1
model.save('model.h5')
