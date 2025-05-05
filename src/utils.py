# src/utils.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import random

SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_h5_data(path):
    with h5py.File(path, 'r') as f:
        X = np.stack([f[key][:] for key in f.keys() if key != 'y'], axis=-1)
        y = f['y'][:].astype(int).flatten() if 'y' in f else None
    return X, y

def plot_signal(signal, title="Señal EEG"):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_conf_matrix(y_true, y_pred, title="Confusion Matrix", cmap='Blues'):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(cmap=cmap, ax=ax)
    plt.title(title)

    # Crear carpeta si no existe
    os.makedirs("results/figures", exist_ok=True)

    # Formatear el nombre del archivo a partir del título
    filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    filepath = os.path.join("results", "figures", filename)

    # Guardar la imagen
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[✓] Imagen guardada en: {filepath}")

    plt.show()
