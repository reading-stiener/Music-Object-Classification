import tensorflow as tf
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

from dataset_reader import class_dataset_reader
import pandas as pd

dataset = class_dataset_reader(records_list="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification",
                                        dest_path= "/home/abi-osler/Documents/CV_final_project/final_project/images_template_matching" )


train_df = pd.read_csv("train_image_annotation.csv", dtype=str)

test_df = pd.read_csv("test_image_annotation.csv", dtype=str) 



model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(220,120,3)))
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
model.add(Dense(118, activation='sigmoid'))

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


train_datagen = ImageDataGenerator(
    rescale=1./ 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, 
    validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="annotation",
        subset="training",
        target_size=(220, 120),
        batch_size=10,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="annotation",
        subset="validation",
        target_size=(220, 120),
        batch_size=10,
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='data/test',
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=10,
        class_mode='categorical')

STEP_SIZE_TRAIN=1000#train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=1000#valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=1000#test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1)
print("answer")
model.evaluate_generator(generator=test_generator,
steps=STEP_SIZE_TEST)