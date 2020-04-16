# Clasificacion de digitos en una imagen con el set MNIST

"""Dataset of 60,000 28x28 grayscale images of the 10 digits, 
along with a test set of 10,000 images.

La clasificacion se realizara con una red neuronal con una capa oculta
que contiene 15 neuronas
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np

# Lectura, visualizacion y pre-procesamiento de los datos
"""
Se retornan 2 tuplas:
- x_train, x_test - array de imagenes en esacala de grises
con dimension (60000,28,28)
- y_train, y_test - array de digitos (0 - 9) con dimension (60000)
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Dimensión de los datos de entrenamiento: {}'.format(x_train.shape[0])) # 60000

# Visualizaremos 16 imagenes aleatorias tomadas del set x_train

# Seleccionamos de la matriz las 16 imagenes aleatorias
ids_imgs = np.random.randint(0, x_train.shape[0],16)
# Las graficamos seleccionando cada una de las imagenes del array
for i in range(len(ids_imgs)):
    img = x_train[ids_imgs[i],:,:]
    plt.subplot(4,4,i+1)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title(y_train[ids_imgs[i]])
plt.suptitle('16 imagenes del set MNIST')
plt.show()

# Preprocesamiento para introducir los datos a la red neuronal
# Se debe aplanar las imagenes en un vector 28x28 = 784 valores
# En este set de datos de 60000 imagenes con dimension 28x28
# shape[0] indica la cantidad de registros - 60000
# shape[1] indica la cantidad de filas - 28
# shape[2] indica la cantidad de columnas - 28

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2])) # (60000, 784)
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_train.shape[2]))  # (10000, 784)

# Normalizamos los valores del vector de imagenes al rango 0-1
# Como las imagenes tienen un rango de 0 a 255, se divide por 255

X_train = X_train/255.0
X_test = X_test/255.0

# Finalmente, convertimos y_train y y_test a representacion "one-hot"
n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

"""
Creacion del modelo
- Capa de entrada: su dimension sera de 784 (el tamano de cada imagen aplanada)
- Capa oculta: 15 neuronas con activacion ReLU
- Capa de salida: funcion de activacion 'softmax' (clasificacion multiclase) y un total
de 10 categorias
"""

#np.random.seed(1)   # Hace que la iteracion inicial siempre empiece en el mismo punto

# Seleccionamos las variables de entrada y salida
input_dim = X_train.shape[1]    # 784 - cantidad de caracteristicas de la matriz de entrada (60000, 784)
output_dim = Y_train.shape[1]   # 10 - cantidad de categorias de la variable de salida (digitos del 0 al 9)

# Definimos la red neuronal

# Creamos el contenedor donde se almacenara
modelo = Sequential()
# Agregamos la capa oculta, cuya dimension en la entrada corresponde a la cantidad de caracteristicas
# de los datos y su dimension de salida el numero de neuronas sobre esa capa
# Se aplica como funcion de activacion relu debido a que no se satura al momento de aplicar
# el gradiente descendente en las iteraciones
modelo.add(Dense(15, input_dim = input_dim, activation = 'relu'))
# Agregamos la capa de salida de la red
# Aplicamos como funcion de activacion la softmax, optima para clasificadores multiclase
# Recordar que esta funcion tiene va de 0 a 1
modelo.add(Dense(output_dim, activation = 'softmax'))
print(modelo.summary())

# Compilacion y entrenamiento, por medio del Gradiente Descendente, un learning rate = 0.05
# funcion de error = entropia cruzada (para evitar minimos locales) y una metrica de desempeno
# basada en la precision

sgd = SGD(lr = 0.2)
modelo.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Para el entrenamiento se usaran 50 iteraciones y un batch_size de 1024
num_epochs = 50
batch_size = 1024
historia = modelo.fit(X_train, Y_train, epochs = num_epochs, batch_size = batch_size, verbose = 2)

#
# Resultados del modelo
#

# Error y precision vs iteraciones
plt.subplot(1,2,1)
plt.plot(historia.history['loss'])
plt.title('Perdida vs iteraciones')
plt.ylabel('Perdida')
plt.xlabel('Iteracion')

plt.subplot(1,2,2)
plt.plot(historia.history['accuracy'])
plt.title('Precision vs iteraciones')
plt.ylabel('Precision')
plt.xlabel('Iteracion')

plt.show()

# Calcular la precision sobre el set de validacion
puntaje = modelo.evaluate(X_test, Y_test, verbose = 0)
print('Precision en el set de validacion: {:.1f}%'.format(100*puntaje[1]))

# Realizar la prediccion sobre el set de validacion y mostrar algunos ejemplos
# de la clasificacion resultante
Y_pred = modelo.predict_classes(X_test)

ids_imgs = np.random.randint(0, X_test.shape[0],9)
for i in range(len(ids_imgs)):
    idx = ids_imgs[i]
    img = X_test[idx,:].reshape(28,28)
    cat_original = np.argmax(Y_test[idx,:])
    cat_prediccion = Y_pred[idx]

    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title('"{}" clasificado como {}'.format(cat_original, cat_prediccion))

plt.suptitle('Ejemplos de clasificacion en el set de validacion')
plt.show()





