import tensorflow as tf
import keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

# Chargement des données MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normaliser les images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255


# Conversion des étiquettes en catégories
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


##### Création du modèle ######
model = Sequential()

# Ajout des couches
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)

# evaluation du modèle
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Perte: {loss}, Précision: {accuracy}')

model.save('mnist_cnn_model.keras')
