from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from dataset_reader import class_dataset_reader
import pandas as pd
import os


# loading dataset
dataset = class_dataset_reader(records_list="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification",
                                        dest_path= "/home/abi-osler/Documents/CV_final_project/final_project/images_template_matching")


#setting up dataframes for batches 
train_df = pd.read_csv("train_image_annotation.csv", dtype=str)
test_df = pd.read_csv("test_image_annotation.csv", dtype=str) 

# traning parameters 
batch_size = 32
num_classes = 118 
epochs = 1
input_shape = (220,120,3)

# directory for saving trained weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_deep_scores_music_object_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

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

opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(optimizer=opt,
             loss="categorical_crossentropy",
             metrics=["accuracy"])


train_datagen = ImageDataGenerator(
    rescale=1./ 255, 
    validation_split=0.10)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="annotation",
        subset="training",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="annotation",
        subset="validation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='data/test',
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

STEP_SIZE_TRAIN=100#train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=100#valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=100#test_generator.n//test_generator.batch_size



checkpoint = ModelCheckpoint(save_dir+"/best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs, 
                    shuffle=True,
                    callbacks=[checkpoint])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Save model and weights

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(generator=test_generator, 
                        steps=STEP_SIZE_TEST)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

