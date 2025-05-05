from src.utils import load_h5_data, SEED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def flatten_signals(X):
    """
    Aplana cada muestra multicanal (128, 9) en un vector de tamaño 128*9.
    """
    return X.reshape((X.shape[0], -1))

def load_flattened_data(path='data/train.h5', n_samples=2000):
    X, y = load_h5_data(path)

    # Submuestreo
    X_small, _, y_small, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=SEED)

    # Aplanar
    X_flat = flatten_signals(X_small)

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    return train_test_split(X_scaled, y_small, test_size=0.2, stratify=y_small, random_state=SEED)
