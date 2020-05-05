import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset_reader import class_dataset_reader

dataset = class_dataset_reader(records_list="/home/abi-osler/Documents/CV_final_project/DeepScoresClassification",
                                        dest_path= "/home/abi-osler/Documents/CV_final_project/final_project/images_template_matching" )
dataset.read_images(train_test=True)

train_set = dataset.train_set() 
test_set = dataset.test_set()


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 220, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(124))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)



#history = model.fit(train_images, train_labels, epochs=10, 
#                    validation_data=(test_images, test_labels))