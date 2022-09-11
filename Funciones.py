import numpy as np
import pandas as pd
import scipy.spatial.distance
from math import *


# Función para primer punto:
def distance_ecu(x_train, x_test_point):

    distances= []  ## create empty list called distances
    for row in range(len(x_train)): ## Loop over the rows of x_train
        current_train_point= x_train[row] #Get them point by point
        current_distance= 0 ## initialize the distance by zero

        for col in range(len(current_train_point)): ## Loop over the columns of the row

            current_distance += (current_train_point[col] - x_test_point[col]) **2
            ## Or current_distance = current_distance + (x_train[i] - x_test_point[i])**2
        current_distance= np.sqrt(current_distance)

        distances.append(current_distance) ## Append the distances

    # Store distances in a dataframe
    distances= pd.DataFrame(data=distances,columns=['dist'])
    return distances

def nearest_neighbors(distance_point, K):

    df_nearest= distance_point.sort_values(by=['dist'], axis=0)

    ## Take only the first K neighbors
    df_nearest= df_nearest[:K]
    return df_nearest

def KNN_from_scratch(x_train, x_test, K):

    distances = []
    ## Loop over all the test set and perform the three steps
    for x_test_point in x_test:
        distance_point  = distance_ecu(x_train, x_test_point)  ## Step 1
        df_nearest_point= nearest_neighbors(distance_point, K)  ## Step 2
        distances.append(df_nearest_point)

    return df_nearest_point

# Una vez se tienen los vectores x1 y x2 se pueden comparar con el módulo de la librería Scipy
# spatial.distance.hamming, el cual recibe los parámetros u, v, w siendo u y v arrays que corresponden a x1 y x2
