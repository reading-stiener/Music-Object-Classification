from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os

# traning parameters 
batch_size = 10
num_classes = 118 
epochs = 5
input_shape = (220,120,3)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(118, activation='softmax'))

# directory for saving trained weights
save_dir = os.path.join(os.getcwd(), 'saved_models2')


# load weights
#model.load_weights(save_dir+ "/best_model.hdf5")
model.load_weights(save_dir+ "/saved-model-908-1.00.hdf5")


# Compile model (required to make predictions)

opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(optimizer=opt,
             loss="categorical_crossentropy",
             metrics=["accuracy"])

print("Created model and loaded weights from file")

# Setting is test data generator 

test_df = pd.read_csv("processed_data_files/test_image_annotation.csv", dtype=str) 

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='data/test',
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

# Score trained model.
scores = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

