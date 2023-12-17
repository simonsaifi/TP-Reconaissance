import tensorflow as tf
import keras as keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('mnist_cnn_model.keras')


(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)

predictions = model.predict(test_images)



# Afficher les premières 10 images, leurs étiquettes prédites, et les vraies étiquettes
for i in range(10):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prédiction : {np.argmax(predictions[i])}, Vrai : {np.argmax(test_labels[i])}")
    plt.show()

loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Perte : {loss}, Précision : {accuracy}")
