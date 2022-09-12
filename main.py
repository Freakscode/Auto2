import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming
from math import *
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer


# Función para primer punto:
def distancia_euclidiana(x_train, x_test_point):
    distancias = []
    for i in range(len(x_train)):
        dis_act_entrenamiento = x_train[i]
        dist_act = 0

        for j in range(len(dis_act_entrenamiento)):
            dist_act += (dis_act_entrenamiento[j] - x_test_point[j]) ** 2

        current_distance = np.sqrt(dist_act)
        distancias.append(current_distance)

    distancias = pd.DataFrame(data=distancias, columns=['dist'])
    return distancias


def vecinos_cercanos(distancia_pp, K):
    vec_cercanos = distancia_pp.sort_values(by=['dist'], axis=0)
    vec_cercanos = vec_cercanos[:K]
    return vec_cercanos


def distancias_knn(x_train, x_test, K):
    distancias = []
    for x_test_point in x_test:
        distancia_pp = distancia_euclidiana(x_train, x_test_point)
        valores_cercanos = vecinos_cercanos(distancia_pp, K)
        distancias.append(valores_cercanos)
        print(f'Valor del punto de prueba: "petal length (cm): {x_test_point[0]}"'
              f' "petal width (cm): {x_test_point[1]}"'
              f' Sus vecinos más cercanos son (ID | Distancia entre los puntos) {valores_cercanos}')
    return


""" Segundo Punto """


def hamming_funct():
    n = int(input(" Ingrese el tamaño del vector: "))
    a = [int]*n
    b = [int]*n
    a = np.random.randint(2, size=n)
    b = np.random.randint(2, size=n)
    hamming_d = hamming(a, b) * len(a)
    print(f'Vector 1: {a} \nVector 2: {b} \nPosiciones en las que fueron diferentes: {hamming_d}')
    return

