

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
import os
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore")

train_datasets = '/Users/toniloperabarril/Desktop/projecte/projecte3/Training'
validation_datasets = '/Users/toniloperabarril/Desktop/projecte/projecte3/Testing'

batch_size = 64
image_size = 256  

def prepare_the_datasets(train_datasets, validation_datasets, batch_size, image_size):
    train_datasets_generator = ImageDataGenerator(rescale=1./255,
                                                  shear_range=0.2, 
                                                  zoom_range=0.2, 
                                                  horizontal_flip=True,
                                                  vertical_flip=True,
                                                  width_shift_range=0.3,
                                                  fill_mode="nearest")

    validation_datasets_generator = ImageDataGenerator(rescale=1.0/255) 
    
    
    train_datasets_generator_data = train_datasets_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_datasets,
        shuffle=True,
        target_size=(image_size, image_size),
        class_mode="categorical"
    )

    validation_datasets_generator_data = validation_datasets_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_datasets,
        shuffle=True,
        target_size=(image_size, image_size),
        class_mode="categorical"
    )

    return train_datasets_generator_data, validation_datasets_generator_data

train_data, validation_data = prepare_the_datasets(train_datasets, validation_datasets, batch_size, image_size)

densenet = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    pooling=None,
    input_shape=(image_size, image_size, 3)
)

for layer in densenet.layers:
    layer.trainable = False

def addTopModel(bottom_model, num_class, D=64):
    top_model = bottom_model.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(D, activation="relu")(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(num_class, activation="softmax")(top_model)
    return top_model

num_classes = 4
Fc_Head = addTopModel(densenet, num_classes)
model = Model(inputs=densenet.input, outputs=Fc_Head)

checkpoint = ModelCheckpoint('/Users/toniloperabarril/Desktop/projecte/model.keras',
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor="val_loss",
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                               factor=0.1,   
                               patience=5,
                               verbose=1,
                               min_delta=0.00001)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

nb_train_samples = 2870
nb_validation_samples = 394


epochs = 50
batch_size = 64

history = model.fit(train_data,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=validation_data,
                    validation_steps=nb_validation_samples // batch_size,
                    shuffle=True)

model_evaluation = model.evaluate(validation_data, batch_size=batch_size)

model.save('/Users/toniloperabarril/Desktop/projecte/model.keras')

def plot_training_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    # Pérdida
    ax1.plot(epochs, loss, label='training_loss', marker='o')
    ax1.plot(epochs, val_loss, label='val_loss', marker='o')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    # Precisión
    ax2.plot(epochs, accuracy, label='training_accuracy', marker='o')
    ax2.plot(epochs, val_accuracy, label='val_accuracy', marker='o')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    plt.show()

plot_training_curves(history)
