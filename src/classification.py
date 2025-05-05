from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_classifiers(X_train, X_test, y_train, y_test, save_path=None, prefix=""):
    results_dir = "results/figures"
    os.makedirs(results_dir, exist_ok=True)

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("=== SVM Report ===")
    print(classification_report(y_test, y_pred_svm))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión - SVM")
    plt.ylabel("Etiqueta verdadera")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{prefix}svm.png"))
    plt.close()

    if save_path:
        # Guardar el modelo SVM
        joblib.dump(svm, save_path)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("=== KNN Report ===")
    print(classification_report(y_test, y_pred_knn))

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens")
    plt.title("Matriz de Confusión - KNN")
    plt.ylabel("Etiqueta verdadera")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{prefix}knn.png"))
    plt.close()

    return svm  # Retornamos el mejor modelo (SVM)