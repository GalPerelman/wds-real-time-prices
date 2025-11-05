import pandas as pd
from sklearn.impute import KNNImputer


def data_completion(data, n_neighbors=3):
    knn_impute = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = knn_impute.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
    return data_imputed