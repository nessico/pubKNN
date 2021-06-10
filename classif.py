import numpy as np
import sklearn
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', lambda x: '%.8f' % x)
dfx = pd.read_csv('~/CSC4850-ML1/csv_files/TrainData5.csv', header = None)
dfy = pd.read_csv('~/CSC4850-ML1/csv_files/TrainLabel5.csv', header = None)

# Sets any NaN input to 0
dfx.values[dfx > 10] = 0

# Load TestData
X_test = pd.read_csv('~/CSC4850-ML1/csv_files/TestData5.csv', header = None)

# Use KNN to determine label values for test input
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(dfx, dfy.values.ravel())
results = knn.predict(X_test)

#Exports results as a csv file
results = pd.DataFrame(results)
results.to_csv('~/CSC4850-ML1/results/KimTrainLabel5.csv', index = False, header = None)
