

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

# Define Inception model
def build_inception_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (1, 1), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Define VGGNet model
def build_vggnet_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Define ResNet model
def build_resnet_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (7, 7), strides=(2, 2), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Compile the models
inception_model = build_inception_model()
vggnet_model = build_vggnet_model()
resnet_model = build_resnet_model()

# Function to train and plot model history
def train_and_plot(model, model_name, history_list):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    history_list.append(history)

    # Plot training history
    plt.plot(history.history['accuracy'], label=f'{model_name} - accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} - val_accuracy')

# Train and plot models
inception_history, vggnet_history, resnet_history = [], [], []

train_and_plot(inception_model, 'Inception', inception_history)
train_and_plot(vggnet_model, 'VGGNet', vggnet_history)
train_and_plot(resnet_model, 'ResNet', resnet_history)

# Plot the comparison graph
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Comparison - Training and Validation Accuracy')
plt.show()


#
#In this code, we have defined three different models: Inception, VGGNet, and ResNet. Each model is defined in a separate function.
#
#We then compile each model using the Adam optimizer, the categorical crossentropy loss function, and the accuracy metric.
#
#The `train_and_plot` function is used to train each model and plot its training history. This function takes a model, its name, and a list to store the training history. It compiles the model, fits the model to the training data, and appends the training history to the provided list.
#
#The function then plots the training accuracy and validation accuracy for the model.
#
#Finally, we train and plot the models. We create separate lists to store the training history for each model. We then call the `train_and_plot` function for each model, passing the model, its name, and the corresponding list to store the training history.
#
#After training and plotting the models, we plot a comparison graph showing the training and validation accuracy for each model.