import h5py
import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import SEED

def prepare_tsfresh_dataframe(X):
    """Convierte una matriz (n_muestras, 128, 9) en formato largo para tsfresh"""
    data_long = []
    for i in range(X.shape[0]):
        for ch in range(X.shape[2]):
            for t in range(X.shape[1]):
                data_long.append([i, t, X[i, t, ch], f"ch{ch}"])
    df_long = pd.DataFrame(data_long, columns=['id', 'time', 'value', 'channel'])
    df_long['channel'] = df_long['channel'].astype('category')
    return df_long

def load_tsfresh_features(path='data/train.h5', n_samples=500):
    with h5py.File(path, 'r') as f:
        X = np.stack([f[key][:] for key in f.keys() if key != 'y'], axis=-1)
        y = f['y'][:].astype(int).flatten()

    X_small, _, y_small, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=SEED)

    # Convertir a DataFrame largo
    data_long = []
    for i in range(X_small.shape[0]):
        for ch in range(X_small.shape[2]):
            for t in range(X_small.shape[1]):
                data_long.append([i, t, X_small[i, t, ch], f"ch{ch}"])
    df_long = pd.DataFrame(data_long, columns=['id', 'time', 'value', 'channel'])
    df_long['channel'] = df_long['channel'].astype('category')

    # Pivotear y extraer características
    df_pivot = df_long.pivot_table(index=['id', 'time'], columns='channel', values='value').reset_index()
    features = extract_features(df_pivot, column_id='id', column_sort='time')
    impute(features)
    features_selected = select_features(features, y_small[:features.shape[0]])

    # Guardar nombres de columnas seleccionadas
    selected_columns = features_selected.columns.tolist()

    # Normalizar
    scaler = StandardScaler()
    X_ts = scaler.fit_transform(features_selected)

    X_train, X_test, y_train, y_test = train_test_split(X_ts, y_small[:features.shape[0]], test_size=0.2, stratify=y_small[:features.shape[0]], random_state=SEED)

    return X_train, X_test, y_train, y_test, selected_columns
