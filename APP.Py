import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Chargement du dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Affichage d'un exemple de chiffre manuscrit
plt.imshow(X_train[0])
plt.show()

# Mise en forme des données pour l'entraînement
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalisation des données
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Conversion des étiquettes en catégories
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Création du modèle CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# Évaluation du modèle
score = model.evaluate(X_test, y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])

#Ce code utilise le dataset MNIST de chiffres manuscrits pour entraîner un modèle de réseau de neurones convolutif (CNN) à reconnaître les chiffres manuscrits.
#Le modèle CNN est créé à l'aide de la bibliothèque Keras, et est ensuite entraîné et évalué sur les données MNIST. 
#Le modèle atteint généralement une précision de classification d'environ 99%.
