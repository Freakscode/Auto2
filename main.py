import numpy as np
import pandas as pd
import scipy.spatial.distance
from math import *


# Función para primer punto:
def distancia_euclidiana(x_train, x_test_point):
    distancias = []  ## create empty list called distances
    for i in range(len(x_train)):  ## Loop over the rows of x_train
        dis_act_entrenamiento = x_train[i]  # Get them point by point
        dist_act = 0  ## initialize the distance by zero

        for j in range(len(dis_act_entrenamiento)):  ## Loop over the columns of the row

            dist_act += (dis_act_entrenamiento[j] - x_test_point[j]) ** 2
            ## Or current_distance = current_distance + (x_train[i] - x_test_point[i])**2
        current_distance = np.sqrt(dist_act)

        distancias.append(current_distance)  ## Append the distances

    # Store distances in a dataframe
    distancias = pd.DataFrame(data=distancias, columns=['dist'])
    return distancias


def vecinos_cercanos(distancia_pp, K):
    vec_cercanos = distancia_pp.sort_values(by=['dist'], axis=0)

    ## Take only the first K neighbors
    vec_cercanos = vec_cercanos[:K]
    return vec_cercanos


def distancias(x_train, x_test, K):
    distancias = []
    ## Loop over all the test set and perform the three steps
    for x_test_point in x_test:
        distancia_pp = distancia_euclidiana(x_train, x_test_point)  ## Step 1
        valores_cercanos = vecinos_cercanos(distancia_pp, K)  ## Step 2
        distancias.append(valores_cercanos)
        print(f'Valor del punto de prueba: "petal length (cm): {x_test_point[0]}"'
              f' "petal width (cm): {x_test_point[1]}"'
              f' Sus vecinos más cercanos son (ID | Distancia entre los puntos) {valores_cercanos}')
    return
