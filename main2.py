import tensorflow as tf
import keras as keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

# Charger le modèle
model = tf.keras.models.load_model('mnist_cnn_model.keras')

# Charger les données MNIST
(_, _), (test_images, test_labels) = mnist.load_data()

# Prétraitement des images
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)

# Faire des prédictions
predictions = model.predict(test_images)

# Afficher des images aléatoires avec les prédictions
num_images = 10
random_indices = np.random.choice(len(test_images), num_images)

for i in random_indices:
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    plt.title(f"Prédiction : {predicted_label}, Vrai : {true_label}")
    plt.show()

# Évaluer la performance du modèle
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Perte : {loss}, Précision : {accuracy}")
