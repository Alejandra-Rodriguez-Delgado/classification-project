import h5py
import numpy as np
import pandas as pd
from joblib import load
import os
from src.preprocessing_flattened import flatten_signals
from src.preprocessing_tsfresh import prepare_tsfresh_dataframe
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import StandardScaler

def generate_submission(use_tsfresh=True, model_path="models/svm_tsfresh.joblib", test_path="data/test.h5", output_path="submission.csv"):
    """
    Genera el archivo submission.csv a partir del modelo entrenado y el archivo test.h5.
    """

    print("\n==============================")
    print("  Generando submission.csv")
    print("==============================")

    # Cargar test.h5
    with h5py.File(test_path, 'r') as f:
        X_test = np.stack([f[key][:] for key in f.keys()], axis=-1)  # No hay etiquetas

    if use_tsfresh:
        # Formatear test set
        df_long = prepare_tsfresh_dataframe(X_test)
        df_pivot = df_long.pivot_table(index=['id', 'time'], columns='channel', values='value').reset_index()

        # Extraer características
        features = extract_features(df_pivot, column_id='id', column_sort='time')
        impute(features)

        # Cargar modelo
        model_bundle = load(model_path)

        model_bundle = load(model_path)

        if isinstance(model_bundle, dict) and "model" in model_bundle and "selected_columns" in model_bundle:
            model = model_bundle["model"]
            selected_columns = model_bundle["selected_columns"]
            features = features[selected_columns]
        else:
            raise ValueError("❌ El modelo no contiene información sobre las columnas seleccionadas.")


        # Escalar
        scaler = StandardScaler()
        X_final = scaler.fit_transform(features)
    else:
        # Señales aplanadas
        model = load(model_path)
        X_flat = flatten_signals(X_test)
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_flat)

    # Predecir
    y_pred = model.predict(X_final)

    # Guardar en CSV
    df_submission = pd.DataFrame({
        "Id": np.arange(len(y_pred)),
        "Label": y_pred
    })
    df_submission.to_csv(output_path, index=False)
    print(f"✅ Archivo guardado como {output_path}")
