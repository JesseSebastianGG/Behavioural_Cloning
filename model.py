### Version warning: unlike lectures I've used Keras 2 (eg fit_generator syntax)

### Imports
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Activation, Lambda, Cropping2D
import matplotlib.pyplot as plt
import random

### Read data
# Near-straight images are ignored to address imbalanced dataset
proportion=0.3
steering_limit=0.5
# Allow eg 30% of low-steering (-0.5 to 0.5) images to be used for training


# Read in CSV file
samples = [] # 'lines' in learn_1
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    count=0
    for line in reader:
        # Ignore header
        if count == 0:
            count += 1
            continue
        # Skip most shots with low steering angles
        elif abs(float(line[3]))<steering_limit and random.random()>proportion:
            count += 1
            continue
        else:
            samples.append(line)
        count += 1

# Label data sets
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

# Prep generator function, including using side images
steer_correction = 0.025
batch_size=32

### Define generator for preprocessing on the fly in small batches
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = [] # 'measurements' in learn_1
            for batch_sample in batch_samples:
                
                name_center = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                name_left = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                left_angle = center_angle + steer_correction
                images.append(left_image)
                angles.append(left_angle)
                
                name_right = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                right_angle = center_angle - steer_correction
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road in model, not here
            images = np.array(images)#[:,60:160,:,:]
            angles = np.array(angles)
            
            # append reflections
            images_reflections = images[:,:,::-1,:]
            angles_reflections = - np.array(angles)
            images = np.concatenate((images,images_reflections),axis=0)
            angles = np.concatenate((angles, angles_reflections),axis=0)
            
            X_train = images
            y_train = angles
            yield sklearn.utils.shuffle(X_train, y_train)

# Prep for keras
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
vert_crop = [60,20] # trim 60 off top, 20 off bottom

### Architecture from NVidia. Credit https://arxiv.org/pdf/1604.07316.pdf
def run_model(out_path='model_6_3.h5'):
    ### Architecture. Credit https://arxiv.org/pdf/1604.07316.pdf

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3), output_shape=(160,320,3)))

    # (1,160,320,3); (1,100,320,3) 1st cropping; (1,80,320,3) 2nd cropping (changes numbers below)
    model.add(Cropping2D(cropping=((vert_crop[0],vert_crop[1]),(0,0))))

    # Conv_1, OUT=24x78x161
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding='valid'))
    model.add(Activation('relu'))
    # Conv_2, OUT=36x37x81
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, padding='valid'))
    model.add(Activation('relu'))
    # Conv_3, OUT=48x16x38
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, padding='valid'))
    model.add(Activation('relu'))

    # Conv_4, OUT=48x14x36
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid'))
    # Conv_5, OUT=48x12x34
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='valid'))

    # Flatten for FC layers; OUT=pi(48,12,34)=19584
    model.add(Flatten())

    # FC_1
    model.add(Dense(units=1164))
    model.add(Activation('relu'))
    # FC_2
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    # FC_3
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    # FC_4
    model.add(Dense(units=10))
    model.add(Activation('relu'))

    # Output
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size,
                        validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
                        epochs=4, verbose = 1)

    model.save(out_path)


if __name__ == '__main__':
    run_model()
