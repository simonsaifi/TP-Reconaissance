import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

# Convertir les prédictions et les vraies étiquettes en classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Générer la matrice de confusion
cm = confusion_matrix(true_classes, predicted_classes)

# Visualisation de la matrice de confusion
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues')
plt.ylabel('Étiquettes Réelles')
plt.xlabel('Étiquettes Prédites')
plt.title('Matrice de Confusion', size = 15)
plt.show()
