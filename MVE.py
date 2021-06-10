# Missing Value Estimation
# Estimated K value by choosing odd numeric estimate of k=sqrt(n), where n is equal to the number of samples

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv('MissingData1.csv', header=None)
df
imputer = KNNImputer(n_neighbors=3)
df_results = imputer.fit_transform(df)
df_results = pd.DataFrame(df_results)
df_results.to_csv('KimKNNData1Results.csv', header=False, index=False)

df = pd.read_csv('MissingData2.csv', header=None)
df
imputer = KNNImputer(n_neighbors=7)
df_results = imputer.fit_transform(df)
df_results = pd.DataFrame(df_results)
df_results.to_csv('KimKNNData2Results.csv', header=False, index=False)

df = pd.read_csv('MissingData3.csv', header=None)
df
imputer = KNNImputer(n_neighbors=9)
df_results = imputer.fit_transform(df)
df_results = pd.DataFrame(df_results)
df_results.to_csv('KimKNNData3Results.csv', header=False, index=False)
