from src.utils import set_seed
from src.preprocessing_flattened import load_flattened_data
from src.preprocessing_tsfresh import load_tsfresh_features
from src.classification import evaluate_classifiers
from src.generate_submission import generate_submission
from joblib import dump 


if __name__ == "__main__":

    set_seed()

    print("\n==============================")
    print("  Clasificación - Señales Aplanadas")
    print("==============================")
    X_train_flat, X_test_flat, y_train_flat, y_test_flat = load_flattened_data()
    model_flat = evaluate_classifiers(
        X_train_flat, X_test_flat, y_train_flat, y_test_flat,
        save_path="models/svm_flattened.joblib",
        prefix="flattened_"
    )

    print("\n==============================")
    print("  Clasificación - tsfresh")
    print("==============================")
    # ⬇️ Captura también las columnas seleccionadas
    X_train_tsfresh, X_test_tsfresh, y_train_tsfresh, y_test_tsfresh, selected_columns_tsfresh = load_tsfresh_features()
    
    model_tsfresh = evaluate_classifiers(
        X_train_tsfresh, X_test_tsfresh, y_train_tsfresh, y_test_tsfresh,
        save_path="models/svm_tsfresh.joblib",
        prefix="tsfresh_"
    )

    # ⬇️ Guarda modelo + columnas seleccionadas en un diccionario
    dump({
        "model": model_tsfresh,
        "selected_columns": selected_columns_tsfresh
    }, "models/svm_tsfresh.joblib")

    # Generar archivo de envío
    generate_submission(use_tsfresh=True, model_path="models/svm_tsfresh.joblib")
