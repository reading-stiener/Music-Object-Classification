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
dataset = class_dataset_reader(data_path="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification", 
                               train_test=True)
                               

dataset.read_images()
#setting up dataframes for batches 
train_df = pd.read_csv("processed_data_files/train_image_annotation.csv", dtype=str)
valid_df = pd.read_csv("processed_data_files/val_image_annotation.csv", dtype=str)
test_df = pd.read_csv("processed_data_files/test_image_annotation.csv", dtype=str) 

# traning parameters 
batch_size = 10
num_classes = 118 
epochs = 100
input_shape = (220,120,3)

# directory for saving trained weights
save_dir = os.path.join(os.getcwd(), 'saved_models1')
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


train_datagen = ImageDataGenerator(rescale=1./ 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filename",
        y_col="annotation",
        target_size=(220, 120),
        batch_size=batch_size,
        class_mode='categorical')

STEP_SIZE_TRAIN=10#train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=10#valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=1000#test_generator.n//test_generator.batch_size



filepath = "saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(save_dir+"/"+filepath, monitor='accuracy', verbose=1, save_best_only=False, mode='max')

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs, 
                    shuffle=True,
                    callbacks=[checkpoint])


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], "g--", label='accuracy')
plt.plot(history.history['val_accuracy'], "b--", label='val_accuracy')
plt.title('Accuracy and Validation accuracy')
plt.ylabel('Data')
plt.xlabel('Epoch')
#plt.xticks(range(1, epochs))
plt.legend(loc='upper left')

plt.savefig("accuracy&validation.png")
plt.show()
plt.close()

# Save model and weights

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

