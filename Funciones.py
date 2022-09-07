import numpy as np
import scipy.spatial.distance


# Creación de vectores X


def vector():
    global x1, x2
    a = int(input("Digite las dimensiones de su vector: "))
    x1 = np.random.randint(2, size=a)
    x2 = np.random.randint(2, size=a)
    print(f'Vector 1: {x1} \nVector 2: {x2}')
    print(type(x1))


# Una vez se tienen los vectores x1 y x2 se pueden comparar con el módulo de la librería Scipy
# spatial.distance.hamming, el cual recibe los parámetros u, v, w siendo u y v arrays que corresponden a x1 y x2

def ed(x1, x2):
    a = scipy.spatial.distance.euclidean(x1, x2) * len(x1)
    print(f'La distancia euclidiana de los vectores {x1} y {x2} es: {a:.0f}')


def funcion_ed():  # Función distancia euclidiana
    vector()
    ed(x1, x2)
